"""CLI di yt_learner: da un URL YouTube a SKILL + scheda KNOWLEDGE.

Esempio:
    python -m yt_learner "https://youtu.be/XXXXXXXXXXX" \
        --brain-dir ./brain/knowledge --skills-dir ./skills --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .brain_writer import find_note_by_source, write_knowledge
from .config import BrainPreset, ConfigError, describe_presets, resolve_brain
from .llm_processor import (
    DEFAULT_MODEL,
    LLMClient,
    LLMError,
    build_knowledge_spec,
    build_skill_spec,
    condense_transcript,
    heuristic_specs,
)
from .skill_builder import build_skill, find_skill_by_source
from .slide_extractor import SlideError, extract_slides
from .utils import dump_json, slugify, write_text
from .yt_extractor import ExtractionError, VideoData, extract

log = logging.getLogger("yt_learner")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


# ------------------------------------------------------------------ argomenti


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-learner",
        description=(
            "Analizza un video tutorial YouTube e genera due artefatti: una SKILL "
            "eseguibile (memoria procedurale) e una scheda KNOWLEDGE per il brain "
            "(memoria dichiarativa)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempi:\n"
            '  yt-learner "https://youtu.be/ABCDEFGHIJK"\n'
            '  yt-learner "https://youtu.be/ABCDEFGHIJK" --model gpt-4o --lang en\n'
            '  yt-learner "https://youtu.be/ABCDEFGHIJK" --brain manent --skills-dir ~/.claude/skills\n'
        ),
    )
    parser.add_argument(
        "url", nargs="?", help="URL (o id) del video YouTube da analizzare"
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--brain",
        help=(
            "destinazione del brain: nome di un preset (default, manent, o definito "
            "in yt-learner.json) oppure un percorso. Vedi --list-brains"
        ),
    )
    output.add_argument(
        "--list-brains",
        action="store_true",
        help="elenca le destinazioni disponibili ed esce",
    )
    output.add_argument(
        "--brain-dir",
        default=None,
        help="cartella esplicita delle schede (override del preset; default: ./brain/knowledge)",
    )
    output.add_argument(
        "--skills-dir",
        default="./skills",
        help="cartella delle skill (default: ./skills)",
    )
    output.add_argument("--skill-name", help="forza il nome della skill (kebab-case)")
    output.add_argument(
        "--only",
        choices=("skill", "knowledge", "both"),
        default="both",
        help="genera solo uno dei due artefatti (default: both)",
    )
    output.add_argument(
        "--index",
        help="file indice del brain (es. MEMORY.md) da aggiornare; default: quello del preset",
    )
    output.add_argument(
        "--no-index",
        action="store_true",
        help="non aggiornare l'indice, anche se il preset ne definisce uno",
    )
    output.add_argument(
        "--no-lint",
        action="store_true",
        help="non lanciare il lint del vault previsto dal preset (es. Manent)",
    )
    output.add_argument(
        "--no-overwrite",
        action="store_true",
        help="non sovrascrivere: crea file/cartelle con suffisso numerico",
    )
    output.add_argument("--dump-json", help="salva estrazione e spec LLM in un file JSON")
    output.add_argument("--transcript-out", help="salva la trascrizione con timestamp in un file")
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="esegue tutto ma non scrive nulla su disco (stampa un'anteprima)",
    )

    llm = parser.add_argument_group("LLM")
    llm.add_argument(
        "--model",
        default=None,
        help=f"modello LLM da usare (default: env YT_LEARNER_MODEL o {DEFAULT_MODEL})",
    )
    llm.add_argument(
        "--provider",
        choices=("gemini", "openai", "anthropic"),
        help="forza il provider (di norma dedotto dal nome del modello)",
    )
    llm.add_argument("--api-key", help="API key esplicita (altrimenti letta dall'ambiente/.env)")
    llm.add_argument(
        "--temperature", type=float, default=0.2, help="temperatura del modello (default: 0.2)"
    )
    llm.add_argument(
        "--max-output-tokens",
        type=int,
        default=32768,
        help="tetto di token in uscita: alzalo se la risposta viene troncata (default: 32768)",
    )
    llm.add_argument(
        "--output-language",
        default="italiano",
        help="lingua dei file generati (default: italiano)",
    )
    llm.add_argument(
        "--offline",
        action="store_true",
        help="nessuna chiamata LLM: genera bozze euristiche dalla trascrizione",
    )

    extraction = parser.add_argument_group("estrazione")
    extraction.add_argument(
        "--lang",
        default="it,en",
        help="lingue dei sottotitoli in ordine di preferenza (default: it,en)",
    )
    extraction.add_argument(
        "--allow-audio-fallback",
        action="store_true",
        help="se mancano i sottotitoli, scarica l'audio e lo trascrive con l'LLM",
    )

    slides = parser.add_argument_group("slide (screenshot reali del video)")
    slides.add_argument(
        "--slides",
        action="store_true",
        help="scarica il video ed estrae le slide con ffmpeg, linkandole nella scheda",
    )
    slides.add_argument(
        "--slides-mode",
        choices=("sample", "timeline", "scene"),
        default="sample",
        help=(
            "sample: campiona il video e prende ogni cambio slide (default); "
            "timeline: una slide per voce della timeline; "
            "scene: stacchi rilevati da ffmpeg (video con tagli netti)"
        ),
    )
    slides.add_argument(
        "--slides-interval",
        type=float,
        default=5.0,
        help="secondi tra due campioni in modalita sample (default: 5)",
    )
    slides.add_argument(
        "--slides-change-bits",
        type=int,
        default=45,
        help=(
            "quanti bit su 256 devono cambiare perche sia una slide nuova "
            "(default: 45; alza se la webcam genera falsi positivi)"
        ),
    )
    slides.add_argument(
        "--slides-dir",
        help="cartella delle immagini (default: <brain-dir>/assets/<slug>)",
    )
    slides.add_argument(
        "--slides-offset",
        type=float,
        default=3.0,
        help="secondi da aggiungere al timestamp per saltare la transizione (default: 3)",
    )
    slides.add_argument(
        "--slides-threshold",
        type=float,
        default=0.25,
        help="sensibilita del rilevamento cambio scena, 0-1 (default: 0.25)",
    )
    slides.add_argument(
        "--slides-min-gap",
        type=float,
        default=4.0,
        help="secondi minimi tra due slide in modalita scene (default: 4)",
    )
    slides.add_argument(
        "--slides-width", type=int, default=1280, help="larghezza in pixel (default: 1280)"
    )
    slides.add_argument(
        "--slides-max-height",
        type=int,
        default=720,
        help="altezza massima del video scaricato (default: 720)",
    )
    slides.add_argument(
        "--max-slides", type=int, default=60, help="tetto di slide estratte (default: 60)"
    )
    slides.add_argument("--keep-video", help="cartella dove tenere il video scaricato")

    parser.add_argument("-v", "--verbose", action="store_true", help="log di debug")
    parser.add_argument("-q", "--quiet", action="store_true", help="stampa solo gli errori")
    parser.add_argument("--version", action="version", version=f"yt-learner {__version__}")
    return parser


# -------------------------------------------------------------------- utility


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _force_utf8_stdout() -> None:
    """Console Windows: senza questo i titoli con accenti escono a pezzi."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # stream gia chiuso o non riconfigurabile
            pass


def _load_dotenv() -> None:
    """Carica il `.env` se `python-dotenv` e installato (opzionale)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _rel_to_cwd(path: Path) -> str:
    """Percorso leggibile per i cross-reference: relativo alla cwd se possibile."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _rel_to_vault(path: Path, preset: BrainPreset) -> str:
    """Riferimento alla scheda da mettere nella skill.

    Se il brain sta dentro il progetto, il percorso relativo alla cwd va bene.
    Se sta altrove (il caso normale: un vault in Documenti), un percorso assoluto
    sarebbe rumore: meglio quello relativo alla radice del vault, che e come le
    note si citano fra loro li dentro.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        pass
    try:
        return resolved.relative_to(preset.root.expanduser().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _say(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message)


def _preview(title: str, spec: dict[str, Any]) -> str:
    import json

    return f"\n--- {title} ---\n" + json.dumps(spec, ensure_ascii=False, indent=2)[:4000]


def _resolve_index(args: argparse.Namespace, preset: BrainPreset) -> Path | None:
    """Indice da aggiornare: flag esplicito, altrimenti quello del preset."""
    if args.no_index:
        return None
    if args.index:
        return Path(args.index).expanduser()
    return preset.index_path


def _run_lint(preset: BrainPreset, *, quiet: bool) -> None:
    """Lint del vault previsto dal preset (es. `manent lint` per un vault Manent)."""
    command = preset.resolved_lint_cmd()
    if not command:
        return
    import subprocess

    _say(f"      lint del vault: {' '.join(command)}", quiet=quiet)
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",   # il lint stampa UTF-8: senza questo su Windows esce a pezzi
            errors="replace",
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[avviso lint] impossibile eseguire il lint: {exc}", file=sys.stderr)
        return

    output = (result.stdout or "") + (result.stderr or "")
    tail = "\n".join(line for line in output.strip().splitlines()[-8:])
    if result.returncode == 0:
        _say(f"      lint OK\n{tail}" if tail else "      lint OK", quiet=quiet)
    else:
        print(
            f"[avviso lint] il vault ha segnalazioni (exit {result.returncode}):\n{tail}",
            file=sys.stderr,
        )


# ------------------------------------------------------------------- pipeline


def _extract_video(args: argparse.Namespace, client: LLMClient | None) -> VideoData:
    languages = [lang.strip() for lang in str(args.lang).split(",") if lang.strip()]
    transcriber = None
    if args.allow_audio_fallback:
        if client is None:
            log.warning("--allow-audio-fallback ignorato in modalita offline.")
        else:
            transcriber = client.transcribe_audio
    return extract(
        args.url,
        languages or ["en"],
        allow_audio_fallback=args.allow_audio_fallback,
        audio_transcriber=transcriber,
    )


def _make_client(args: argparse.Namespace) -> LLMClient | None:
    if args.offline:
        return None
    import os

    model = args.model or os.getenv("YT_LEARNER_MODEL") or DEFAULT_MODEL
    return LLMClient(
        model=model,
        provider=args.provider,
        api_key=args.api_key,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )


def run(argv: Sequence[str] | None = None) -> int:
    """Entry point programmatico: ritorna il codice di uscita."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _force_utf8_stdout()
    _setup_logging(args.verbose, args.quiet)
    _load_dotenv()
    quiet = args.quiet

    if args.list_brains:
        print(describe_presets())
        return EXIT_OK
    if not args.url:
        parser.error("serve l'URL del video (oppure usa --list-brains)")

    # destinazione del brain: preset built-in, preset utente o percorso esplicito
    try:
        preset = resolve_brain(args.brain, args.brain_dir)
    except ConfigError as exc:
        print(f"[errore configurazione] {exc}", file=sys.stderr)
        return EXIT_USAGE

    # 1. client LLM (fallisce subito se manca la API key, prima di scaricare tutto)
    try:
        client = _make_client(args)
    except LLMError as exc:
        print(f"[errore LLM] {exc}", file=sys.stderr)
        print("Suggerimento: usa --offline per generare bozze senza LLM.", file=sys.stderr)
        return EXIT_USAGE

    # 2. estrazione video
    try:
        _say(f"[1/4] Estrazione video: {args.url}", quiet=quiet)
        video = _extract_video(args, client)
    except ExtractionError as exc:
        print(f"[errore estrazione] {exc}", file=sys.stderr)
        if "trascrizione" in str(exc).lower():
            print(
                "Suggerimento: prova --lang en (o un'altra lingua) oppure "
                "--allow-audio-fallback per trascrivere l'audio con l'LLM.",
                file=sys.stderr,
            )
        return EXIT_ERROR

    meta = video.metadata
    _say(
        f"      titolo: {meta.title or 'n/d'}\n"
        f"      canale: {meta.channel or 'n/d'} | durata: {meta.duration_human} | "
        f"parole: {video.word_count} | fonte: {video.transcript_source}",
        quiet=quiet,
    )

    transcript = video.timestamped_text()
    # scritta anche in dry-run: e' un output chiesto esplicitamente (modalita "Claude scrive dal transcript")
    if args.transcript_out:
        path = write_text(Path(args.transcript_out), transcript)
        _say(f"      trascrizione salvata in {path}", quiet=quiet)

    # 3. spec via LLM (o euristica offline)
    want_skill = args.only in ("skill", "both")
    want_knowledge = args.only in ("knowledge", "both")
    skill_spec: dict[str, Any] | None = None
    knowledge_spec: dict[str, Any] | None = None

    try:
        if client is None:
            _say("[2/4] Modalita offline: genero bozze euristiche (niente LLM).", quiet=quiet)
            offline_skill, offline_knowledge = heuristic_specs(video)
            skill_spec = offline_skill if want_skill else None
            knowledge_spec = offline_knowledge if want_knowledge else None
        else:
            condensed = condense_transcript(
                client, transcript, progress=lambda msg: _say(f"      {msg}", quiet=quiet)
            )
            if want_skill:
                _say(f"[2/4] Genero la SKILL con {client.provider}/{client.model}...", quiet=quiet)
                skill_spec = build_skill_spec(
                    client, video, condensed, output_language=args.output_language
                )
            if want_knowledge:
                _say(f"[3/4] Genero la KNOWLEDGE con {client.provider}/{client.model}...", quiet=quiet)
                knowledge_spec = build_knowledge_spec(
                    client, video, condensed, output_language=args.output_language
                )
    except LLMError as exc:
        print(f"[errore LLM] {exc}", file=sys.stderr)
        print("Suggerimento: riprova, cambia --model o usa --offline.", file=sys.stderr)
        return EXIT_ERROR

    # 4. percorsi e cross-reference (calcolati prima di scrivere, per linkarsi a vicenda)
    learned = date.today().isoformat()
    skills_dir = Path(args.skills_dir).expanduser()
    brain_dir = preset.target_dir.expanduser()

    # idempotenza: se questo video e gia stato appreso, si riusano i nomi di
    # allora. Il modello propone slug e nome skill diversi a ogni giro, quindi
    # senza questo un secondo passaggio creerebbe doppioni invece di aggiornare.
    existing_skill = find_skill_by_source(skills_dir, meta.url) if want_skill else None
    existing_note = find_note_by_source(brain_dir, meta.url) if want_knowledge else None
    if existing_skill and not args.skill_name:
        log.info("Skill gia esistente per questo video: aggiorno %s", existing_skill)
    if existing_note:
        log.info("Scheda gia esistente per questo video: aggiorno %s", existing_note.name)

    skill_name = slugify(
        args.skill_name
        or existing_skill
        or (skill_spec or {}).get("skill_name")
        or meta.title
        or "skill"
    )
    # lo slug della scheda viene dal TITOLO del video, non dalla proposta del
    # modello: stesso video -> stesso file, anche a distanza di giorni
    knowledge_slug = slugify(meta.title or (knowledge_spec or {}).get("title") or meta.video_id)
    note_filename = existing_note.name if existing_note else preset.filename(knowledge_slug, learned)
    skill_rel = _rel_to_cwd(skills_dir / skill_name / "SKILL.md") if want_skill else None
    knowledge_rel = _rel_to_vault(brain_dir / note_filename, preset) if want_knowledge else None

    if args.dry_run:
        _say(f"[4/4] Dry-run: nessun file scritto (destinazione: {brain_dir}).", quiet=quiet)
        if skill_spec:
            _say(_preview("SKILL SPEC", skill_spec), quiet=quiet)
        if knowledge_spec:
            _say(_preview("KNOWLEDGE SPEC", knowledge_spec), quiet=quiet)
        return EXIT_OK

    # 4b. slide: screenshot reali del video ai punti della timeline
    slides: list[dict[str, Any]] = []
    if args.slides:
        slides_dir = (
            Path(args.slides_dir).expanduser()
            if args.slides_dir
            else brain_dir / "assets" / knowledge_slug
        )
        try:
            _say(
                f"      estraggo le slide ({args.slides_mode}) — scarico il video, "
                "puo richiedere qualche minuto...",
                quiet=quiet,
            )
            extracted = extract_slides(
                meta.video_id,
                slides_dir,
                mode=args.slides_mode,
                timeline=(knowledge_spec or {}).get("timeline") or [],
                max_height=args.slides_max_height,
                width=args.slides_width,
                offset=args.slides_offset,
                threshold=args.slides_threshold,
                min_gap=args.slides_min_gap,
                max_slides=args.max_slides,
                interval=args.slides_interval,
                change_bits=args.slides_change_bits,
                keep_video=Path(args.keep_video).expanduser() if args.keep_video else None,
            )
            for slide in extracted:
                try:
                    relative = slide.path.resolve().relative_to(brain_dir.resolve()).as_posix()
                except ValueError:
                    relative = slide.path.resolve().as_posix()
                slides.append(
                    {"seconds": slide.seconds, "path": relative, "label": slide.label}
                )
            _say(f"      {len(slides)} slide salvate in {slides_dir}", quiet=quiet)
        except SlideError as exc:
            # le slide sono un extra: non devono far fallire skill e knowledge
            print(f"[avviso slide] {exc}", file=sys.stderr)

    # 5. scrittura
    written: list[str] = []
    skill_result = None
    if skill_spec is not None:
        skill_result = build_skill(
            skill_spec,
            video,
            skills_dir,
            skill_name=skill_name,
            knowledge_rel_path=knowledge_rel,
            learned_date=learned,
            overwrite=not args.no_overwrite,
        )
        written.append(f"SKILL      -> {skill_result.skill_file}")
        written += [f"  asset    -> {asset}" for asset in skill_result.assets]

    knowledge_result = None
    if knowledge_spec is not None:
        index_path = _resolve_index(args, preset)
        knowledge_result = write_knowledge(
            knowledge_spec,
            video,
            brain_dir,
            skill_rel_path=skill_rel,
            skill_name=skill_result.skill_name if skill_result else None,
            learned_date=learned,
            overwrite=not args.no_overwrite,
            index_path=index_path,
            filename=note_filename,
            note_type=preset.note_type,
            legacy_metadata_type=preset.legacy_metadata_type,
            slides=slides,
        )
        written.append(f"KNOWLEDGE  -> {knowledge_result.path}  [brain: {preset.name}]")
        if slides:
            written.append(f"  slide    -> {len(slides)} immagini")
        if knowledge_result.index_updated and index_path:
            written.append(f"  indice   -> {index_path}")

    if args.dump_json:
        payload = {
            "video": meta.to_dict(),
            "transcript_source": video.transcript_source,
            "transcript_language": video.transcript_language,
            "skill_spec": skill_spec,
            "knowledge_spec": knowledge_spec,
        }
        path = dump_json(Path(args.dump_json), payload)
        written.append(f"JSON       -> {path}")

    _say("[4/4] Fatto:\n" + "\n".join(written), quiet=quiet)

    if knowledge_result is not None and not args.no_lint:
        _run_lint(preset, quiet=quiet)
    return EXIT_OK


def main() -> None:
    """Entry point da console script."""
    try:
        sys.exit(run())
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
