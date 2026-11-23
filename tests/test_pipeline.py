"""Test offline della pipeline: nessuna rete, nessuna API key.

Esecuzione:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date

from yt_learner import (
    brain_writer,
    config,
    llm_processor,
    main as cli,
    skill_builder,
    slide_extractor,
    utils,
)
from yt_learner.yt_extractor import (
    TranscriptSegment,
    VideoData,
    VideoMetadata,
    _parse_json3,
    _parse_vtt,
    _pick_caption_track,
    extract_video_id,
    InvalidURLError,
)


def fake_video() -> VideoData:
    metadata = VideoMetadata(
        video_id="ABCDEFGHIJK",
        url="https://www.youtube.com/watch?v=ABCDEFGHIJK",
        title="Costruire un Agente AI in Python",
        channel="Canale Test",
        duration_seconds=930,
        upload_date="2026-02-10",
        description="Tutorial su agenti, tool calling e memoria.",
        tags=["ai", "python", "agents"],
    )
    segments = [
        TranscriptSegment(0, 15, "Benvenuti, oggi costruiamo un agente in Python."),
        TranscriptSegment(35, 20, "Installiamo le dipendenze con pip install openai."),
        TranscriptSegment(120, 25, "Definiamo il tool di ricerca e lo registriamo."),
        TranscriptSegment(300, 30, "Gestiamo l'errore di rate limit con un retry."),
        TranscriptSegment(600, 30, "Validiamo eseguendo lo script e leggendo l'output."),
    ]
    return VideoData(metadata, segments, transcript_source="test", transcript_language="it")


class TestUtils(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(utils.slugify("Come costruire un Agente AI! (2026)"), "come-costruire-un-agente-ai-2026")
        self.assertEqual(utils.slugify("Però àccentì"), "pero-accenti")
        self.assertEqual(utils.slugify(""), "untitled")
        self.assertLessEqual(len(utils.slugify("parola " * 40)), 60)

    def test_timestamps(self):
        self.assertEqual(utils.fmt_timestamp(135), "02:15")
        self.assertEqual(utils.fmt_timestamp(3725), "01:02:05")
        self.assertEqual(utils.parse_timestamp("02:15"), 135)
        self.assertEqual(utils.parse_timestamp("[01:02:03]"), 3723)
        self.assertEqual(utils.parse_timestamp("boh"), 0)

    def test_frontmatter_quoting(self):
        block = utils.yaml_frontmatter({"description": "titolo: sottotitolo", "tags": ["a", "b"], "vuoto": None})
        self.assertIn('description: "titolo: sottotitolo"', block)
        self.assertIn("tags: [a, b]", block)
        self.assertNotIn("vuoto", block)


class TestExtractorParsing(unittest.TestCase):
    def test_video_id(self):
        for url in (
            "https://www.youtube.com/watch?v=ABCDEFGHIJK",
            "https://youtu.be/ABCDEFGHIJK?t=30",
            "https://www.youtube.com/shorts/ABCDEFGHIJK",
            "ABCDEFGHIJK",
        ):
            self.assertEqual(extract_video_id(url), "ABCDEFGHIJK")
        with self.assertRaises(InvalidURLError):
            extract_video_id("https://example.com/video")

    def test_vtt_dedup(self):
        vtt = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:04.000\nciao <c>mondo</c>\n\n"
            "00:00:04.000 --> 00:00:07.000\nciao mondo\n\n"
            "00:00:07.000 --> 00:00:09.500\nsecondo blocco\n"
        )
        segments = _parse_vtt(vtt)
        self.assertEqual([s.text for s in segments], ["ciao mondo", "secondo blocco"])
        self.assertEqual(segments[1].timestamp, "00:07")

    def test_json3(self):
        payload = json.dumps(
            {"events": [{"tStartMs": 2000, "dDurationMs": 1500, "segs": [{"utf8": "hello "}, {"utf8": "world"}]}]}
        )
        segments = _parse_json3(payload)
        self.assertEqual(segments[0].text, "hello world")
        self.assertEqual(segments[0].timestamp, "00:02")

    def test_caption_track_preference(self):
        info = {
            "subtitles": {},
            "automatic_captions": {
                "en": [{"ext": "vtt", "url": "u-en"}],
                "it-IT": [{"ext": "json3", "url": "u-it"}],
            },
        }
        self.assertEqual(_pick_caption_track(info, ["it", "en"]), ("u-it", "it-IT"))
        self.assertIsNone(_pick_caption_track({"subtitles": {}}, ["en"]))

    def test_timestamped_text(self):
        text = fake_video().timestamped_text(60)
        self.assertTrue(text.startswith("[00:00]"))
        self.assertIn("[02:00]", text)


class TestLLMProcessor(unittest.TestCase):
    def test_json_parsing_tollerante(self):
        self.assertEqual(llm_processor.parse_json_response('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(llm_processor.parse_json_response('Ecco:\n{"a": {"b": "}"}}\nfine'), {"a": {"b": "}"}})
        with self.assertRaises(ValueError):
            llm_processor.parse_json_response("nessun json")

    def test_detect_provider(self):
        self.assertEqual(llm_processor.detect_provider("gemini-2.5-flash"), "gemini")
        self.assertEqual(llm_processor.detect_provider("gpt-4o"), "openai")
        self.assertEqual(llm_processor.detect_provider("claude-sonnet-5"), "anthropic")
        with self.assertRaises(llm_processor.LLMError):
            llm_processor.detect_provider("llama3")

    def test_normalizza_spec_sporca(self):
        """L'LLM restituisce tipi sbagliati: la normalizzazione non deve rompersi."""
        video = fake_video()
        spec = llm_processor.normalize_skill_spec(
            {
                "skill_name": "Agente AI Python!",
                "tags": "python",
                "workflow": ["fai la cosa", {"title": "Secondo", "commands": "pip install x"}],
                "error_handling": ["rate limit"],
                "prerequisites": ["assunzione sciolta"],
                "assets": [{"path": "../fuori.py", "content": "x"}, {"path": "scripts/ok.py", "content": "print(1)"}],
            },
            video,
        )
        self.assertEqual(spec["skill_name"], "agente-ai-python")
        self.assertEqual(spec["tags"], ["python"])
        self.assertEqual(spec["workflow"][0]["step"], 1)
        self.assertEqual(spec["workflow"][1]["commands"], ["pip install x"])
        self.assertEqual(spec["error_handling"][0]["error"], "rate limit")
        self.assertEqual(spec["prerequisites"]["assumptions"], ["assunzione sciolta"])
        self.assertEqual(len(spec["assets"]), 2)  # il filtro sul path avviene in build_skill

    def test_timestamp_inventati_scartati(self):
        video = fake_video()  # durata 930s
        spec = llm_processor.normalize_knowledge_spec(
            {
                "title": "T",
                "timeline": [
                    {"timestamp": "05:00", "label": "reale"},
                    {"timestamp": "99:00", "label": "inventato"},
                    {"timestamp": "01:00", "label": "primo"},
                ],
            },
            video,
        )
        self.assertEqual([e["label"] for e in spec["timeline"]], ["primo", "reale"])

    def test_heuristic_specs(self):
        skill_spec, knowledge_spec = llm_processor.heuristic_specs(fake_video())
        self.assertTrue(skill_spec["workflow"])
        self.assertTrue(knowledge_spec["timeline"])
        self.assertEqual(knowledge_spec["slug"], "costruire-un-agente-ai-in-python")

    def test_condense_senza_client_tronca(self):
        lungo = "riga\n" * 40000
        condensato = llm_processor.condense_transcript(None, lungo)
        self.assertLess(len(condensato), len(lungo))
        self.assertIn("troncata", condensato)

    def test_chunking_rispetta_le_righe(self):
        chunks = llm_processor._chunk_text("\n".join(f"riga {i}" for i in range(1000)), size=200)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("\n".join(chunks), "\n".join(f"riga {i}" for i in range(1000)))


class TestLLMFlowConMock(unittest.TestCase):
    """Percorso LLM completo, con `generate_text` finto: nessuna rete."""

    def _client(self, responses):
        client = llm_processor.LLMClient.__new__(llm_processor.LLMClient)
        client.model, client.provider, client.temperature = "fake-model", "gemini", 0.2
        client.max_output_tokens, client.api_key, client._client = 1024, "x", object()
        client.generate_text = mock.Mock(side_effect=responses)
        return client

    def test_build_skill_spec(self):
        payload = json.dumps(
            {
                "skill_name": "Costruire Agente!",
                "description": "come si fa",
                "workflow": [{"step": 1, "title": "Setup", "commands": ["pip install openai"]}],
                "validation": [{"check": "run", "how": "python main.py", "success_criteria": "exit 0"}],
            }
        )
        client = self._client([payload])
        spec = llm_processor.build_skill_spec(client, fake_video(), "[00:00] testo")
        self.assertEqual(spec["skill_name"], "costruire-agente")
        self.assertEqual(spec["workflow"][0]["commands"], ["pip install openai"])
        # il prompt contiene contesto video, trascrizione e schema
        prompt = client.generate_text.call_args.args[1]
        self.assertIn("Costruire un Agente AI in Python", prompt)
        self.assertIn("[00:00] testo", prompt)
        self.assertIn('"workflow"', prompt)

    def test_retry_su_json_invalido(self):
        client = self._client(["non e json", json.dumps({"title": "Ok", "timeline": []})])
        spec = llm_processor.build_knowledge_spec(client, fake_video(), "[00:00] testo")
        self.assertEqual(spec["title"], "Ok")
        self.assertEqual(client.generate_text.call_count, 2)
        self.assertIn("JSON valido", client.generate_text.call_args.args[1])

    def test_errore_dopo_troppi_retry(self):
        client = self._client(["nope"] * 3)
        with self.assertRaises(llm_processor.LLMError):
            llm_processor.build_skill_spec(client, fake_video(), "x")

    def test_condense_map_reduce(self):
        lungo = "\n".join(f"[00:{i:02d}] riga molto lunga {'x' * 500}" for i in range(400))
        self.assertGreater(len(lungo), llm_processor.MAX_PROMPT_CHARS)
        client = self._client(["- nota sintetica"] * 20)
        condensato = llm_processor.condense_transcript(client, lungo)
        self.assertGreater(client.generate_text.call_count, 1)
        self.assertIn("nota sintetica", condensato)
        self.assertLess(len(condensato), len(lungo))


class TestBuilders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.video = fake_video()
        self.skill_spec, self.knowledge_spec = llm_processor.heuristic_specs(self.video)

    def test_build_skill_scrive_md_e_asset(self):
        self.skill_spec["assets"] = [
            {"path": "scripts/run.py", "language": "python", "purpose": "demo", "content": "print('ok')"},
            {"path": "../evasione.py", "language": "python", "purpose": "cattivo", "content": "boom"},
            {"path": "scripts/malware.exe", "language": "bin", "purpose": "cattivo", "content": "boom"},
        ]
        result = skill_builder.build_skill(
            self.skill_spec, self.video, self.root / "skills", knowledge_rel_path="brain/knowledge/x.md"
        )
        content = result.skill_file.read_text(encoding="utf-8")
        self.assertTrue(result.skill_file.exists())
        self.assertIn("name: costruire-un-agente-ai-in-python", content)
        self.assertIn("## Workflow passo-passo", content)
        self.assertIn("## Criteri di validazione", content)
        self.assertIn("brain/knowledge/x.md", content)
        # solo l'asset legittimo viene scritto
        self.assertEqual([p.name for p in result.assets], ["run.py"])
        self.assertFalse((self.root / "evasione.py").exists())

    def test_no_overwrite_crea_suffisso(self):
        skills = self.root / "skills"
        first = skill_builder.build_skill(self.skill_spec, self.video, skills)
        second = skill_builder.build_skill(self.skill_spec, self.video, skills, overwrite=False)
        self.assertNotEqual(first.skill_dir, second.skill_dir)
        self.assertTrue(second.skill_name.endswith("-2"))

    def test_write_knowledge_e_indice(self):
        brain = self.root / "brain" / "knowledge"
        index = self.root / "brain" / "MEMORY.md"
        result = brain_writer.write_knowledge(
            self.knowledge_spec,
            self.video,
            brain,
            skill_rel_path="skills/mia-skill/SKILL.md",
            skill_name="mia-skill",
            index_path=index,
        )
        content = result.path.read_text(encoding="utf-8")
        self.assertIn("source_url: https://www.youtube.com/watch?v=ABCDEFGHIJK", content)
        self.assertIn("related_skill: skills/mia-skill/SKILL.md", content)
        self.assertIn("## Timeline / segnalibri", content)
        self.assertIn("&t=", content)  # deep link col timestamp
        self.assertIn("[[mia-skill]]", content)
        self.assertTrue(result.index_updated)
        index_text = index.read_text(encoding="utf-8")
        self.assertIn("knowledge/costruire-un-agente-ai-in-python.md", index_text)

        # riscrivere non duplica la riga di indice
        brain_writer.write_knowledge(self.knowledge_spec, self.video, brain, index_path=index)
        self.assertEqual(index.read_text(encoding="utf-8").count("costruire-un-agente-ai-in-python.md"), 1)


class TestBrainPresets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_preset_manent(self):
        with mock.patch.dict("os.environ", {"MANENT_VAULT": str(self.root / "vault")}):
            preset = config.resolve_brain("manent")
        self.assertEqual(preset.target_dir, self.root / "vault" / "library")
        self.assertEqual(preset.note_type, "raw-source")
        self.assertFalse(preset.legacy_metadata_type)
        self.assertEqual(preset.index_path, self.root / "vault" / "MEMORY.md")
        self.assertTrue(preset.filename("mio-video").startswith(date.today().isoformat()))

    def test_preset_default_e_percorso(self):
        self.assertEqual(config.resolve_brain().target_dir, Path("brain/knowledge"))
        diretto = config.resolve_brain(str(self.root / "altrove"))
        self.assertEqual(diretto.target_dir, self.root / "altrove")
        self.assertEqual(diretto.filename("x"), "x.md")

    def test_brain_dir_batte_il_preset_ma_ne_eredita_le_convenzioni(self):
        with mock.patch.dict("os.environ", {"MANENT_VAULT": str(self.root / "vault")}):
            preset = config.resolve_brain("manent", brain_dir=str(self.root / "fuori"))
        self.assertEqual(preset.target_dir, self.root / "fuori")
        self.assertEqual(preset.note_type, "raw-source")   # convenzione conservata
        self.assertIsNone(preset.index_path)               # indice no: cartella diversa

    def test_riferimento_alla_scheda_relativo_al_vault(self):
        """La skill cita la scheda come la citerebbe una nota del vault."""
        vault = self.root / "vault"
        preset = config.resolve_brain(str(vault))
        preset.root, preset.subdir = vault, "library"
        nota = vault / "library" / "2026-08-20-video.md"
        self.assertEqual(cli._rel_to_vault(nota, preset), "library/2026-08-20-video.md")

        # brain dentro il progetto: resta relativo alla cwd
        with mock.patch.object(Path, "cwd", return_value=self.root):
            dentro = config.resolve_brain(str(self.root / "brain"))
            self.assertEqual(
                cli._rel_to_vault(self.root / "brain" / "x.md", dentro), "brain/x.md"
            )

    def test_preset_sconosciuto(self):
        with self.assertRaises(config.ConfigError):
            config.resolve_brain("inesistente")

    def test_preset_da_file_utente(self):
        (self.root / "yt-learner.json").write_text(
            json.dumps({"brains": {"lavoro": {"root": str(self.root / "v"), "subdir": "fonti",
                                              "layout": "dated", "note_type": "raw-source",
                                              "index": "INDEX.md"}}}),
            encoding="utf-8",
        )
        with mock.patch.object(Path, "cwd", return_value=self.root):
            preset = config.resolve_brain("lavoro")
        self.assertEqual(preset.target_dir, self.root / "v" / "fonti")
        self.assertEqual(preset.index_path, self.root / "v" / "INDEX.md")


class TestSlides(unittest.TestCase):
    """Funzioni pure dell'estrattore slide: nessun ffmpeg, nessun video."""

    def test_dhash_e_hamming(self):
        gradiente = bytes((value * 7) % 256 for value in range(17 * 16))
        piatto = bytes([128] * (17 * 16))
        self.assertNotEqual(slide_extractor._dhash(gradiente), slide_extractor._dhash(piatto))
        self.assertEqual(slide_extractor._hamming(0b1011, 0b1110), 2)

    def test_detail_distingue_frame_vuoto(self):
        self.assertEqual(slide_extractor._thumb_detail(bytes([200] * 72)), 0.0)
        self.assertGreater(slide_extractor._thumb_detail(bytes(range(72))), 10)

    def test_dedup(self):
        self.assertTrue(slide_extractor._is_duplicate(0b1010, [0b1011], threshold=6))
        self.assertFalse(slide_extractor._is_duplicate(0xFFFF, [0x0000], threshold=6))
        self.assertFalse(slide_extractor._is_duplicate(None, [1, 2], threshold=6))

    def test_ffmpeg_mancante_messaggio_chiaro(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(slide_extractor.SlideError) as ctx:
                slide_extractor.ffmpeg_path()
        self.assertIn("ffmpeg", str(ctx.exception))

    def test_match_slide_su_timeline(self):
        timeline = [{"seconds": 0}, {"seconds": 100}, {"seconds": 400}]
        slides = [
            {"seconds": 5, "path": "a.jpg"},     # illustra la voce 0
            {"seconds": 60, "path": "b.jpg"},    # avanzo (voce 0 gia servita)
            {"seconds": 105, "path": "c.jpg"},   # illustra la voce 100
            {"seconds": 900, "path": "d.jpg"},   # fuori finestra: avanzo
        ]
        matched, leftover = brain_writer._match_slides_to_timeline(timeline, slides)
        self.assertEqual(matched[0]["path"], "a.jpg")
        self.assertEqual(matched[100]["path"], "c.jpg")
        self.assertNotIn(400, matched)
        self.assertEqual([s["path"] for s in leftover], ["b.jpg", "d.jpg"])

    def test_scheda_con_slide(self):
        video = fake_video()
        spec = llm_processor.heuristic_specs(video)[1]
        markdown = brain_writer.render_knowledge_markdown(
            spec,
            video,
            slides=[
                {"seconds": 0, "path": "assets/x/00000-slide.jpg", "label": "intro"},
                {"seconds": 9999, "path": "assets/x/09999-slide.jpg", "label": "extra"},
            ],
        )
        self.assertIn("slides: 2", markdown)
        self.assertIn("![slide 00:00](assets/x/00000-slide.jpg)", markdown)
        self.assertIn("## Slide estratte dal video", markdown)

    def test_note_type_e_nome_dal_preset(self):
        video = fake_video()
        spec = llm_processor.heuristic_specs(video)[1]
        markdown = brain_writer.render_knowledge_markdown(
            spec, video, note_type="raw-source", note_name="2026-08-20-video", legacy_metadata_type=True
        )
        self.assertIn("name: 2026-08-20-video", markdown)
        self.assertIn("type: raw-source", markdown)
        self.assertIn("  type: raw-source", markdown)  # metadata.type legacy


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _run(self, extra):
        argv = [
            "https://youtu.be/ABCDEFGHIJK",
            "--offline",
            "--brain-dir", str(self.root / "brain" / "knowledge"),
            "--skills-dir", str(self.root / "skills"),
            "--quiet",
        ] + extra
        with mock.patch.object(cli, "extract", return_value=fake_video()):
            return cli.run(argv)

    def test_end_to_end_offline(self):
        code = self._run(["--dump-json", str(self.root / "out.json")])
        self.assertEqual(code, 0)
        skill = self.root / "skills" / "costruire-un-agente-ai-in-python" / "SKILL.md"
        note = self.root / "brain" / "knowledge" / "costruire-un-agente-ai-in-python.md"
        self.assertTrue(skill.exists())
        self.assertTrue(note.exists())
        # cross-reference bidirezionale
        self.assertIn("costruire-un-agente-ai-in-python.md", skill.read_text(encoding="utf-8"))
        self.assertIn("SKILL.md", note.read_text(encoding="utf-8"))
        payload = json.loads((self.root / "out.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["video"]["video_id"], "ABCDEFGHIJK")

    def test_only_skill_e_nome_forzato(self):
        code = self._run(["--only", "skill", "--skill-name", "Mia Skill Custom"])
        self.assertEqual(code, 0)
        self.assertTrue((self.root / "skills" / "mia-skill-custom" / "SKILL.md").exists())
        self.assertFalse((self.root / "brain").exists())

    def test_rieseguire_aggiorna_invece_di_duplicare(self):
        """Stesso video due volte: stessi file, anche se l'LLM cambia i nomi."""
        self.assertEqual(self._run([]), 0)
        skill_dir = self.root / "skills" / "costruire-un-agente-ai-in-python"
        note = self.root / "brain" / "knowledge" / "costruire-un-agente-ai-in-python.md"
        self.assertTrue(skill_dir.exists() and note.exists())

        # secondo giro: il "modello" propone nomi completamente diversi
        originale = llm_processor.heuristic_specs

        def altri_nomi(video):
            skill_spec, knowledge_spec = originale(video)
            skill_spec["skill_name"] = "nome-inventato-al-secondo-giro"
            knowledge_spec["slug"] = "slug-inventato-al-secondo-giro"
            return skill_spec, knowledge_spec

        with mock.patch.object(cli, "heuristic_specs", altri_nomi):
            self.assertEqual(self._run([]), 0)

        skills = sorted(p.name for p in (self.root / "skills").iterdir())
        note_files = sorted(p.name for p in (self.root / "brain" / "knowledge").iterdir())
        self.assertEqual(skills, ["costruire-un-agente-ai-in-python"])
        self.assertEqual(note_files, ["costruire-un-agente-ai-in-python.md"])

    def test_riusa_il_nome_datato_gia_esistente(self):
        """Con layout datato, il giorno dopo si aggiorna il file di ieri."""
        brain = self.root / "vault" / "library"
        brain.mkdir(parents=True)
        vecchia = brain / "2026-08-01-video-di-ieri.md"
        vecchia.write_text(
            "---\nname: 2026-08-01-video-di-ieri\nsource_url: https://www.youtube.com/watch?v=ABCDEFGHIJK\n---\n",
            encoding="utf-8",
        )
        trovata = brain_writer.find_note_by_source(
            brain, "https://www.youtube.com/watch?v=ABCDEFGHIJK"
        )
        self.assertEqual(trovata, vecchia)
        self.assertIsNone(brain_writer.find_note_by_source(brain, "https://youtu.be/altro"))

    def test_trova_la_skill_dal_video(self):
        skills = self.root / "skills" / "mia-skill"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text(
            "---\nname: mia-skill\nsource_url: https://www.youtube.com/watch?v=ABCDEFGHIJK\n---\n",
            encoding="utf-8",
        )
        self.assertEqual(
            skill_builder.find_skill_by_source(
                self.root / "skills", "https://www.youtube.com/watch?v=ABCDEFGHIJK"
            ),
            "mia-skill",
        )
        self.assertIsNone(
            skill_builder.find_skill_by_source(self.root / "skills", "https://youtu.be/altro")
        )

    def test_dry_run_non_scrive(self):
        self.assertEqual(self._run(["--dry-run"]), 0)
        self.assertFalse((self.root / "skills").exists())
        self.assertFalse((self.root / "brain").exists())

    def test_errore_estrazione_exit_1(self):
        from yt_learner.yt_extractor import TranscriptUnavailableError

        argv = ["https://youtu.be/ABCDEFGHIJK", "--offline", "--quiet"]
        with mock.patch.object(cli, "extract", side_effect=TranscriptUnavailableError("niente sottotitoli")):
            self.assertEqual(cli.run(argv), 1)

    def test_api_key_mancante_exit_2(self):
        argv = ["https://youtu.be/ABCDEFGHIJK", "--model", "gpt-4o"]
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            with mock.patch.object(cli, "_load_dotenv", lambda: None):
                self.assertEqual(cli.run(argv), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
