"""Strato LLM: trasforma la trascrizione in due spec JSON (skill + knowledge).

Provider supportati (scelti in automatico dal nome del modello, oppure con
`--provider`):
  - `gemini`     -> pacchetto `google-genai`   (GEMINI_API_KEY / GOOGLE_API_KEY)
  - `openai`     -> pacchetto `openai`         (OPENAI_API_KEY)
  - `anthropic`  -> pacchetto `anthropic`      (ANTHROPIC_API_KEY)

Esiste anche una modalita `--offline` (nessuna chiamata di rete) che produce
spec euristiche dai soli metadati/trascrizione: utile per test e per non
bloccare la pipeline quando manca la API key.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .utils import fmt_timestamp, parse_timestamp, slugify, snippet, truncate
from .yt_extractor import TranscriptSegment, VideoData

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"

# Budget di caratteri per singolo prompt: oltre questa soglia la trascrizione
# viene condensata con un passaggio map-reduce.
MAX_PROMPT_CHARS = 120_000
CHUNK_CHARS = 40_000


class LLMError(RuntimeError):
    """Errore di configurazione o di chiamata verso il provider LLM."""


# --------------------------------------------------------------- provider glue


def detect_provider(model: str) -> str:
    """Deduce il provider dal nome del modello."""
    name = (model or "").lower()
    if name.startswith(("gemini", "models/gemini", "gemma")):
        return "gemini"
    if name.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if name.startswith("claude"):
        return "anthropic"
    raise LLMError(
        f"Provider non deducibile dal modello {model!r}. "
        "Passa --provider gemini|openai|anthropic."
    )


_ENV_KEYS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


def _resolve_api_key(provider: str, api_key: str | None) -> str:
    if api_key:
        return api_key
    for env_var in _ENV_KEYS.get(provider, ()):
        value = os.getenv(env_var)
        if value:
            return value
    raise LLMError(
        f"API key mancante per il provider {provider!r}. "
        f"Esporta una di queste variabili: {', '.join(_ENV_KEYS.get(provider, ()))} "
        "(oppure mettila in un file .env)."
    )


# ------------------------------------------------------------------ JSON utils


def _strip_code_fences(text: str) -> str:
    """Rimuove eventuali fence markdown attorno al JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    return cleaned


def _extract_json_object(text: str) -> str:
    """Isola il primo oggetto JSON bilanciato presente nel testo."""
    start = text.find("{")
    if start == -1:
        raise ValueError("Nessun oggetto JSON trovato nella risposta.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("Oggetto JSON incompleto nella risposta (parentesi non bilanciate).")


def parse_json_response(text: str) -> dict[str, Any]:
    """Parsing tollerante della risposta LLM (fence, testo attorno, ecc.)."""
    if not text or not text.strip():
        raise ValueError("Risposta LLM vuota.")
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(_extract_json_object(cleaned))


# ------------------------------------------------------------------- il client


TRUNCATED_REASONS = {"max_tokens", "maxtokens", "length"}


def _gemini_finish_reason(response: Any) -> str | None:
    """Motivo di chiusura della prima candidate, se il SDK lo espone."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    return getattr(reason, "name", None) or (str(reason) if reason else None)


def _raise_if_truncated(finish_reason: str | None, budget: int) -> None:
    """Una risposta tagliata a meta non e un problema di formato: e un budget.

    Senza questo controllo il retry rimanda lo stesso prompt con lo stesso
    tetto di token e fallisce identico, tre volte, dicendo solo "JSON non
    valido".
    """
    if not finish_reason:
        return
    if finish_reason.strip().lower().lstrip("finish_reason.") in TRUNCATED_REASONS:
        raise LLMError(
            f"Risposta troncata dal limite di output ({budget} token). "
            "Rilancia con --max-output-tokens piu alto, oppure con --model piu capiente."
        )


class LLMClient:
    """Wrapper minimale e uniforme sui tre provider supportati."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        provider: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        # 8k non bastano: su un tutorial da mezz'ora la spec della skill viene
        # troncata a meta e il JSON arriva con le parentesi non chiuse
        max_output_tokens: int = 32768,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.provider = (provider or detect_provider(self.model)).lower()
        if self.provider not in _ENV_KEYS:
            raise LLMError(f"Provider non supportato: {self.provider!r}.")
        self.api_key = _resolve_api_key(self.provider, api_key)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = None  # istanziato pigramente al primo uso

    # -- costruzione client -------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "gemini":
            try:
                from google import genai
            except ImportError as exc:
                raise LLMError("Pacchetto mancante: pip install google-genai") from exc
            self._client = genai.Client(api_key=self.api_key)
        elif self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("Pacchetto mancante: pip install openai") from exc
            self._client = OpenAI(api_key=self.api_key)
        else:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError("Pacchetto mancante: pip install anthropic") from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    # -- generazione --------------------------------------------------------

    def generate_text(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        """Una singola chiamata di completamento; ritorna il testo grezzo."""
        client = self._get_client()
        try:
            if self.provider == "gemini":
                config: dict[str, Any] = {
                    "system_instruction": system_prompt,
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }
                if json_mode:
                    config["response_mime_type"] = "application/json"
                response = client.models.generate_content(
                    model=self.model, contents=user_prompt, config=config
                )
                _raise_if_truncated(_gemini_finish_reason(response), self.max_output_tokens)
                return response.text or ""

            if self.provider == "openai":
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                # i modelli di reasoning (o1/o3/o4) non accettano temperature
                if not self.model.lower().startswith(("o1", "o3", "o4")):
                    kwargs["temperature"] = self.temperature
                response = client.chat.completions.create(**kwargs)
                _raise_if_truncated(
                    getattr(response.choices[0], "finish_reason", None), self.max_output_tokens
                )
                return response.choices[0].message.content or ""

            # anthropic
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Chiamata a {self.provider}/{self.model} fallita: {exc}") from exc

    def generate_json(
        self, system_prompt: str, user_prompt: str, *, retries: int = 2
    ) -> dict[str, Any]:
        """Come `generate_text` ma con parsing JSON e retry sul formato."""
        prompt = user_prompt
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            raw = self.generate_text(system_prompt, prompt, json_mode=True)
            try:
                return parse_json_response(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                log.warning("Risposta non-JSON (tentativo %s/%s): %s", attempt + 1, retries + 1, exc)
                prompt = (
                    f"{user_prompt}\n\n"
                    "ATTENZIONE: la risposta precedente non era JSON valido "
                    f"({exc}). Rispondi SOLO con un oggetto JSON valido, senza testo "
                    "attorno e senza fence markdown."
                )
        raise LLMError(f"Il modello non ha prodotto JSON valido dopo {retries + 1} tentativi: {last_error}")

    # -- audio --------------------------------------------------------------

    def transcribe_audio(self, audio_path: Path) -> list[TranscriptSegment]:
        """Trascrive un file audio (fallback quando mancano i sottotitoli)."""
        audio_path = Path(audio_path)
        if self.provider == "openai":
            return self._transcribe_openai(audio_path)
        if self.provider == "gemini":
            return self._transcribe_gemini(audio_path)
        raise LLMError(
            "Il provider anthropic non supporta la trascrizione audio: "
            "usa --provider openai o gemini per --allow-audio-fallback."
        )

    def _transcribe_openai(self, audio_path: Path) -> list[TranscriptSegment]:
        client = self._get_client()
        model = "whisper-1"
        try:
            with audio_path.open("rb") as handle:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=handle,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
        except Exception as exc:
            raise LLMError(f"Trascrizione OpenAI ({model}) fallita: {exc}") from exc

        payload = response if isinstance(response, dict) else response.model_dump()
        segments = []
        for item in payload.get("segments") or []:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            start = float(item.get("start") or 0)
            end = float(item.get("end") or start)
            segments.append(TranscriptSegment(start, max(0.0, end - start), text))
        if not segments and payload.get("text"):
            segments = [TranscriptSegment(0.0, 0.0, str(payload["text"]).strip())]
        return segments

    def _transcribe_gemini(self, audio_path: Path) -> list[TranscriptSegment]:
        client = self._get_client()
        try:
            uploaded = client.files.upload(file=str(audio_path))
            # l'upload e asincrono: aspettiamo che il file diventi ACTIVE
            for _ in range(60):
                state = getattr(getattr(uploaded, "state", None), "name", str(getattr(uploaded, "state", "")))
                if state in ("ACTIVE", ""):
                    break
                if state == "FAILED":
                    raise LLMError("Gemini non e riuscito a processare il file audio caricato.")
                time.sleep(2)
                uploaded = client.files.get(name=uploaded.name)

            prompt = (
                "Trascrivi integralmente l'audio. Rispondi SOLO con JSON: "
                '{"segments": [{"start_seconds": <float>, "text": "<testo>"}]}. '
                "Crea un segmento ogni 15-30 secondi di parlato, con lo start reale."
            )
            response = client.models.generate_content(
                model=self.model,
                contents=[uploaded, prompt],
                config={"response_mime_type": "application/json", "temperature": 0.0},
            )
            data = parse_json_response(response.text or "")
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Trascrizione Gemini fallita: {exc}") from exc

        segments = []
        for item in data.get("segments") or []:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            start = item.get("start_seconds", item.get("start", 0))
            try:
                start_value = float(start)
            except (TypeError, ValueError):
                start_value = float(parse_timestamp(str(start)))
            segments.append(TranscriptSegment(start_value, 0.0, text))
        return segments


# ----------------------------------------------------------------- i prompt


SKILL_SYSTEM_PROMPT = """Sei un ingegnere di agenti AI. Il tuo compito e convertire il
tutorial video fornito in una SKILL ESEGUIBILE (memoria procedurale) che un agente
autonomo possa seguire senza aver visto il video.

Regole non negoziabili:
- Ogni passo deve essere azionabile, sequenziale e deterministico: niente "guarda il video",
  niente "come mostrato a schermo".
- Estrai comandi di shell e codice REALI citati nel tutorial. Se il video li mostra solo a
  schermo e la trascrizione e ambigua, ricostruisci lo snippet minimo corretto e segnalalo
  nel campo `explanation` con il prefisso "(ricostruito)".
- Non inventare API, flag o nomi di libreria che non esistono. Se un dettaglio manca,
  scrivilo esplicitamente come assunzione o come passo di verifica.
- I criteri di validazione devono essere osservabili (output di un comando, file creato,
  status code, test che passa).
- Rispondi SOLO con JSON valido conforme allo schema richiesto."""

SKILL_SCHEMA_HINT = """{
  "skill_name": "kebab-case, max 5 parole, descrive il TASK non il video",
  "description": "una riga: cosa fa la skill e quando usarla",
  "version": "0.1.0",
  "tags": ["3-8 tag tecnici"],
  "triggers": ["frasi/intenti utente che devono attivare la skill"],
  "objective": "risultato finale che l'agente ottiene eseguendo la skill",
  "prerequisites": {
    "tools": ["binari/CLI necessari con versione minima se citata"],
    "libraries": ["pacchetti con versione se citata"],
    "credentials": ["API key / variabili d'ambiente necessarie"],
    "assumptions": ["assunzioni fatte dove il video non e esplicito"]
  },
  "workflow": [
    {
      "step": 1,
      "title": "titolo breve del passo",
      "instructions": "istruzioni operative dettagliate per l'agente",
      "commands": ["comandi shell da eseguire, uno per elemento"],
      "expected_result": "cosa deve osservare l'agente se il passo e riuscito",
      "timestamp": "MM:SS del video da cui viene il passo"
    }
  ],
  "code_snippets": [
    {
      "title": "titolo",
      "language": "python|bash|json|yaml|...",
      "filename": "percorso suggerito, es. agent.py (opzionale)",
      "code": "codice completo e pronto all'uso",
      "explanation": "cosa fa e dove si inserisce nel workflow"
    }
  ],
  "error_handling": [
    {"error": "messaggio/sintomo", "cause": "causa", "fix": "soluzione operativa"}
  ],
  "validation": [
    {"check": "cosa verificare", "how": "comando o azione di verifica", "success_criteria": "output atteso"}
  ],
  "assets": [
    {
      "path": "scripts/nome.py oppure examples/nome.json",
      "language": "python|bash|json|...",
      "purpose": "a cosa serve il file",
      "content": "contenuto completo del file"
    }
  ],
  "references": ["link o risorse citate nel video"]
}"""

KNOWLEDGE_SYSTEM_PROMPT = """Sei il curatore di una knowledge base (brain) di un sistema
agentico. Converti il tutorial video in una scheda di memoria dichiarativa/episodica:
serve a RICORDARE il video (di cosa parlava, quando dice cosa, perche e rilevante), non a
eseguirlo passo passo.

Regole:
- I timestamp della timeline devono essere REALI, presi dai marcatori [MM:SS] della
  trascrizione fornita. Non inventarli mai.
- La sintesi deve essere comprensibile a chi non ha visto il video.
- Distingui i concetti (teoria, architettura, trade-off) dalle istruzioni operative:
  le istruzioni finiscono nella skill, qui restano i principi.
- Rispondi SOLO con JSON valido conforme allo schema richiesto."""

KNOWLEDGE_SCHEMA_HINT = """{
  "title": "titolo leggibile della scheda",
  "slug": "kebab-case per il nome file",
  "one_liner": "una riga che riassume il video",
  "executive_summary": "3-6 frasi: cosa spiega il video e perche e rilevante",
  "relevance": "perche questa conoscenza e utile a un sistema agentico / quando richiamarla",
  "key_concepts": [
    {"concept": "nome del concetto", "explanation": "spiegazione in 2-4 frasi", "timestamp": "MM:SS"}
  ],
  "architecture": "descrizione dell'architettura o del flusso illustrato (stringa, anche vuota)",
  "tools_mentioned": ["tool, librerie, servizi citati"],
  "timeline": [
    {"timestamp": "MM:SS", "label": "titolo del segmento", "detail": "1-2 frasi di sintesi"}
  ],
  "takeaways": ["insegnamenti azionabili, 3-7 elementi"],
  "pitfalls": ["errori/trappole citate nel video"],
  "open_questions": ["cosa il video non spiega e andrebbe approfondito"],
  "tags": ["3-8 tag"]
}"""

CONDENSE_SYSTEM_PROMPT = """Comprimi la porzione di trascrizione mantenendo TUTTI i dettagli
tecnici: comandi, nomi di file, librerie, versioni, parametri, errori citati e relative
soluzioni. Mantieni i marcatori [MM:SS] all'inizio di ogni blocco che sintetizzi.
Rispondi con note dense in elenco puntato, senza preamboli."""


# ------------------------------------------------------- condensazione (map-reduce)


def _chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Divide il testo in blocchi rispettando i confini di riga."""
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        if current_len + len(line) + 1 > size and current:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def condense_transcript(
    client: LLMClient | None, transcript: str, *, progress: Callable[[str], None] | None = None
) -> str:
    """Riduce una trascrizione troppo lunga con un passaggio map-reduce.

    Senza client (modalita offline) fa un troncamento onesto e dichiarato.
    """
    if len(transcript) <= MAX_PROMPT_CHARS:
        return transcript
    if client is None:
        return truncate(transcript, MAX_PROMPT_CHARS)

    chunks = _chunk_text(transcript)
    notes: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if progress:
            progress(f"condenso blocco {index}/{len(chunks)} della trascrizione")
        summary = client.generate_text(
            CONDENSE_SYSTEM_PROMPT,
            f"Blocco {index}/{len(chunks)} della trascrizione:\n\n{chunk}",
        )
        notes.append(summary.strip())
    condensed = "\n\n".join(notes)
    # se anche le note restano enormi, tronchiamo dichiarandolo
    return truncate(condensed, MAX_PROMPT_CHARS)


# ------------------------------------------------------------------ contesto


def _video_context(video: VideoData) -> str:
    """Blocco di contesto (metadati) premesso a ogni prompt."""
    meta = video.metadata
    description = truncate(meta.description or "", 2000)
    return (
        f"TITOLO: {meta.title}\n"
        f"CANALE: {meta.channel}\n"
        f"URL: {meta.url}\n"
        f"DURATA: {meta.duration_human}\n"
        f"DATA PUBBLICAZIONE: {meta.upload_date or 'n/d'}\n"
        f"LINGUA TRASCRIZIONE: {video.transcript_language or 'n/d'}\n"
        f"FONTE TRASCRIZIONE: {video.transcript_source}\n"
        f"TAG YOUTUBE: {', '.join(meta.tags) if meta.tags else 'n/d'}\n"
        f"DESCRIZIONE:\n{description or 'n/d'}\n"
    )


# --------------------------------------------------------------- normalizzazione


def _as_list(value: Any) -> list[Any]:
    """Accetta liste, stringhe o None e ritorna sempre una lista pulita."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item not in (None, "")]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [value]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


def normalize_skill_spec(spec: dict[str, Any], video: VideoData) -> dict[str, Any]:
    """Riempie i buchi e uniforma i tipi: i builder non devono mai fare KeyError."""
    spec = dict(spec or {})
    fallback_name = slugify(spec.get("skill_name") or video.metadata.title or "skill-da-video")
    prerequisites = spec.get("prerequisites") or {}
    if not isinstance(prerequisites, dict):
        prerequisites = {"assumptions": _as_list(prerequisites)}

    workflow: list[dict[str, Any]] = []
    for index, step in enumerate(_as_list(spec.get("workflow")), start=1):
        if not isinstance(step, dict):
            step = {"instructions": _as_str(step)}
        workflow.append(
            {
                "step": step.get("step") or index,
                "title": _as_str(step.get("title")) or f"Passo {index}",
                "instructions": _as_str(step.get("instructions")),
                "commands": [_as_str(c) for c in _as_list(step.get("commands"))],
                "expected_result": _as_str(step.get("expected_result")),
                "timestamp": _as_str(step.get("timestamp")),
            }
        )

    snippets = []
    for snippet in _as_list(spec.get("code_snippets")):
        if not isinstance(snippet, dict):
            continue
        snippets.append(
            {
                "title": _as_str(snippet.get("title")) or "Snippet",
                "language": _as_str(snippet.get("language")) or "text",
                "filename": _as_str(snippet.get("filename")),
                "code": _as_str(snippet.get("code")),
                "explanation": _as_str(snippet.get("explanation")),
            }
        )

    errors = []
    for item in _as_list(spec.get("error_handling")):
        if isinstance(item, dict):
            errors.append(
                {
                    "error": _as_str(item.get("error")),
                    "cause": _as_str(item.get("cause")),
                    "fix": _as_str(item.get("fix")),
                }
            )
        else:
            errors.append({"error": _as_str(item), "cause": "", "fix": ""})

    validation = []
    for item in _as_list(spec.get("validation")):
        if isinstance(item, dict):
            validation.append(
                {
                    "check": _as_str(item.get("check")),
                    "how": _as_str(item.get("how")),
                    "success_criteria": _as_str(item.get("success_criteria")),
                }
            )
        else:
            validation.append({"check": _as_str(item), "how": "", "success_criteria": ""})

    assets = []
    for item in _as_list(spec.get("assets")):
        if not isinstance(item, dict):
            continue
        path = _as_str(item.get("path")).strip().lstrip("/\\")
        content = _as_str(item.get("content"))
        if not path or not content:
            continue
        assets.append(
            {
                "path": path,
                "language": _as_str(item.get("language")) or "text",
                "purpose": _as_str(item.get("purpose")),
                "content": content,
            }
        )

    return {
        "skill_name": fallback_name,
        "description": _as_str(spec.get("description")) or f"Skill derivata da: {video.metadata.title}",
        "version": _as_str(spec.get("version")) or "0.1.0",
        "tags": [_as_str(t) for t in _as_list(spec.get("tags"))],
        "triggers": [_as_str(t) for t in _as_list(spec.get("triggers"))],
        "objective": _as_str(spec.get("objective")),
        "prerequisites": {
            "tools": [_as_str(t) for t in _as_list(prerequisites.get("tools"))],
            "libraries": [_as_str(t) for t in _as_list(prerequisites.get("libraries"))],
            "credentials": [_as_str(t) for t in _as_list(prerequisites.get("credentials"))],
            "assumptions": [_as_str(t) for t in _as_list(prerequisites.get("assumptions"))],
        },
        "workflow": workflow,
        "code_snippets": snippets,
        "error_handling": errors,
        "validation": validation,
        "assets": assets,
        "references": [_as_str(r) for r in _as_list(spec.get("references"))],
    }


def normalize_knowledge_spec(spec: dict[str, Any], video: VideoData) -> dict[str, Any]:
    """Come sopra, per la scheda di knowledge; normalizza anche i timestamp."""
    spec = dict(spec or {})
    duration = video.metadata.duration_seconds or 0

    timeline = []
    for item in _as_list(spec.get("timeline")):
        if not isinstance(item, dict):
            continue
        raw_ts = _as_str(item.get("timestamp"))
        seconds = parse_timestamp(raw_ts)
        # scarta timestamp palesemente inventati (oltre la durata + 1 min di margine)
        if duration and seconds > duration + 60:
            log.warning("Timestamp %s fuori dalla durata del video: lo scarto.", raw_ts)
            continue
        timeline.append(
            {
                "timestamp": fmt_timestamp(seconds),
                "seconds": seconds,
                "label": _as_str(item.get("label")) or "Segmento",
                "detail": _as_str(item.get("detail")),
            }
        )
    timeline.sort(key=lambda entry: entry["seconds"])

    concepts = []
    for item in _as_list(spec.get("key_concepts")):
        if isinstance(item, dict):
            concepts.append(
                {
                    "concept": _as_str(item.get("concept")) or "Concetto",
                    "explanation": _as_str(item.get("explanation")),
                    "timestamp": _as_str(item.get("timestamp")),
                }
            )
        else:
            concepts.append({"concept": _as_str(item), "explanation": "", "timestamp": ""})

    title = _as_str(spec.get("title")) or video.metadata.title or "Video senza titolo"
    return {
        "title": title,
        "slug": slugify(_as_str(spec.get("slug")) or title),
        "one_liner": _as_str(spec.get("one_liner")),
        "executive_summary": _as_str(spec.get("executive_summary")),
        "relevance": _as_str(spec.get("relevance")),
        "key_concepts": concepts,
        "architecture": _as_str(spec.get("architecture")),
        "tools_mentioned": [_as_str(t) for t in _as_list(spec.get("tools_mentioned"))],
        "timeline": timeline,
        "takeaways": [_as_str(t) for t in _as_list(spec.get("takeaways"))],
        "pitfalls": [_as_str(t) for t in _as_list(spec.get("pitfalls"))],
        "open_questions": [_as_str(t) for t in _as_list(spec.get("open_questions"))],
        "tags": [_as_str(t) for t in _as_list(spec.get("tags"))],
    }


# ------------------------------------------------------------- API principale


def build_skill_spec(
    client: LLMClient,
    video: VideoData,
    transcript: str,
    *,
    output_language: str = "italiano",
) -> dict[str, Any]:
    """Chiede all'LLM la spec della skill e la normalizza."""
    user_prompt = (
        f"{_video_context(video)}\n"
        f"Lingua di output: {output_language}.\n\n"
        "TRASCRIZIONE (con marcatori temporali):\n"
        f"{transcript}\n\n"
        "Genera la SKILL seguendo ESATTAMENTE questo schema JSON:\n"
        f"{SKILL_SCHEMA_HINT}\n\n"
        "Popola `assets` solo se il video mostra script o template abbastanza "
        "corposi da meritare un file a parte."
    )
    spec = client.generate_json(SKILL_SYSTEM_PROMPT, user_prompt)
    return normalize_skill_spec(spec, video)


def build_knowledge_spec(
    client: LLMClient,
    video: VideoData,
    transcript: str,
    *,
    output_language: str = "italiano",
) -> dict[str, Any]:
    """Chiede all'LLM la spec della scheda knowledge e la normalizza."""
    user_prompt = (
        f"{_video_context(video)}\n"
        f"Lingua di output: {output_language}.\n\n"
        "TRASCRIZIONE (con marcatori temporali):\n"
        f"{transcript}\n\n"
        "Genera la scheda KNOWLEDGE seguendo ESATTAMENTE questo schema JSON:\n"
        f"{KNOWLEDGE_SCHEMA_HINT}\n\n"
        "La timeline deve avere tra 5 e 15 voci, con timestamp presi dai marcatori "
        "[MM:SS] della trascrizione."
    )
    spec = client.generate_json(KNOWLEDGE_SYSTEM_PROMPT, user_prompt)
    return normalize_knowledge_spec(spec, video)


# --------------------------------------------------------- modalita offline


def _split_segments(segments: Sequence[TranscriptSegment], parts: int) -> list[list[TranscriptSegment]]:
    if not segments:
        return []
    size = max(1, len(segments) // max(1, parts))
    return [list(segments[i : i + size]) for i in range(0, len(segments), size)][:parts]


def heuristic_specs(video: VideoData) -> tuple[dict[str, Any], dict[str, Any]]:
    """Spec di fallback senza LLM: struttura corretta, contenuto grezzo.

    Serve a due cose: testare la pipeline senza API key e non perdere il lavoro
    di estrazione quando il provider e irraggiungibile. I file generati sono
    marcati esplicitamente come da rivedere a mano.
    """
    meta = video.metadata
    title = meta.title or f"Video {meta.video_id}"
    note = "(modalita offline: contenuto estratto meccanicamente, da rivedere)"

    blocks = _split_segments(video.segments, 8)
    timeline = [
        {
            "timestamp": fmt_timestamp(block[0].start),
            "seconds": int(block[0].start),
            "label": " ".join(block[0].text.split()[:8]) or "Segmento",
            "detail": snippet(" ".join(s.text for s in block[:4]), 280),
        }
        for block in blocks
        if block
    ]

    skill_spec = normalize_skill_spec(
        {
            "skill_name": slugify(title),
            "description": f"Skill bozza da '{title}' {note}",
            "objective": f"Riprodurre la procedura mostrata in '{title}'.",
            "tags": meta.tags[:6],
            "triggers": [title],
            "prerequisites": {"assumptions": ["Prerequisiti non estratti: modalita offline."]},
            "workflow": [
                {
                    "step": index,
                    "title": entry["label"],
                    "instructions": entry["detail"],
                    "timestamp": entry["timestamp"],
                }
                for index, entry in enumerate(timeline, start=1)
            ],
            "validation": [
                {
                    "check": "Revisione umana",
                    "how": "Rileggere la skill confrontandola col video",
                    "success_criteria": "I passi sono eseguibili senza guardare il video",
                }
            ],
            "references": [meta.url],
        },
        video,
    )

    knowledge_spec = normalize_knowledge_spec(
        {
            "title": title,
            "slug": slugify(title),
            "one_liner": snippet(video.plain_text, 180),
            "executive_summary": truncate(video.plain_text, 1200),
            "relevance": f"Appunti grezzi dal video del canale {meta.channel or 'n/d'} {note}",
            "timeline": timeline,
            "tools_mentioned": meta.tags[:10],
            "tags": meta.tags[:8],
            "open_questions": ["Rigenerare la scheda con un LLM per una sintesi vera."],
        },
        video,
    )
    return skill_spec, knowledge_spec
