"""Estrazione da YouTube: metadati + trascrizione, con fallback a cascata.

Ordine dei tentativi per la trascrizione:
  1. `youtube-transcript-api`  (sottotitoli manuali, poi automatici)
  2. `yt-dlp`                  (sottotitoli/auto-caption scaricati e parsati)
  3. audio + LLM multimodale   (opzionale, solo con --allow-audio-fallback)

Tutti gli errori "di dominio" derivano da `ExtractionError`, cosi la CLI puo
stampare un messaggio comprensibile invece di uno stacktrace.
"""

from __future__ import annotations

import io
import json
import logging
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .utils import fmt_timestamp

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; yt-learner/0.1)"


# ------------------------------------------------------------------- eccezioni


class ExtractionError(RuntimeError):
    """Errore generico della pipeline di estrazione."""


class InvalidURLError(ExtractionError):
    """L'URL passato non contiene un video id YouTube riconoscibile."""


class VideoUnavailableError(ExtractionError):
    """Video privato, rimosso, con restrizioni geografiche o age-gated."""


class TranscriptUnavailableError(ExtractionError):
    """Nessuna trascrizione recuperabile con le strategie disponibili."""


# ----------------------------------------------------------------- data models


@dataclass
class VideoMetadata:
    """Metadati del video, normalizzati indipendentemente dalla fonte."""

    video_id: str
    url: str
    title: str = ""
    channel: str = ""
    channel_url: str = ""
    duration_seconds: int = 0
    upload_date: str = ""          # ISO `YYYY-MM-DD` quando disponibile
    description: str = ""
    tags: list[str] = field(default_factory=list)
    view_count: int | None = None
    thumbnail: str = ""
    source: str = "yt-dlp"         # da dove arrivano questi metadati

    @property
    def duration_human(self) -> str:
        return fmt_timestamp(self.duration_seconds) if self.duration_seconds else "n/d"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_human"] = self.duration_human
        return data


@dataclass
class TranscriptSegment:
    """Un segmento di trascrizione con il suo offset in secondi."""

    start: float
    duration: float
    text: str

    @property
    def timestamp(self) -> str:
        return fmt_timestamp(self.start)


@dataclass
class VideoData:
    """Risultato completo dell'estrazione: metadati + trascrizione."""

    metadata: VideoMetadata
    segments: list[TranscriptSegment] = field(default_factory=list)
    transcript_source: str = ""
    transcript_language: str = ""

    @property
    def plain_text(self) -> str:
        """Trascrizione come testo continuo (senza timestamp)."""
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def timestamped_text(self, bucket_seconds: int = 30) -> str:
        """Trascrizione raggruppata in blocchi con timestamp.

        Serve all'LLM per poter citare `[MM:SS]` reali nella timeline del brain.
        """
        if not self.segments:
            return ""
        lines: list[str] = []
        bucket_start = self.segments[0].start
        buffer: list[str] = []
        for segment in self.segments:
            text = segment.text.strip()
            if not text:
                continue
            if segment.start - bucket_start >= bucket_seconds and buffer:
                lines.append(f"[{fmt_timestamp(bucket_start)}] {' '.join(buffer)}")
                buffer, bucket_start = [], segment.start
            buffer.append(text)
        if buffer:
            lines.append(f"[{fmt_timestamp(bucket_start)}] {' '.join(buffer)}")
        return "\n".join(lines)

    @property
    def word_count(self) -> int:
        return len(self.plain_text.split())


# -------------------------------------------------------------------- video id

_ID_PATTERNS = (
    r"(?:v=|/v/|/embed/|/shorts/|/live/|youtu\.be/)([0-9A-Za-z_-]{11})",
    r"^([0-9A-Za-z_-]{11})$",
)


def extract_video_id(url: str) -> str:
    """Ricava l'id da qualsiasi forma di URL YouTube (o dall'id stesso)."""
    if not url or not url.strip():
        raise InvalidURLError("URL vuoto.")
    candidate = url.strip()
    for pattern in _ID_PATTERNS:
        match = re.search(pattern, candidate)
        if match:
            return match.group(1)
    raise InvalidURLError(
        f"Impossibile estrarre il video id da: {url!r}. Formati attesi: "
        "https://www.youtube.com/watch?v=XXXXXXXXXXX oppure https://youtu.be/XXXXXXXXXXX"
    )


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# -------------------------------------------------------------------- metadati


def _normalize_upload_date(raw: Any) -> str:
    """`20260115` (formato yt-dlp) -> `2026-01-15`."""
    text = str(raw or "")
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _metadata_from_ytdlp(video_id: str) -> tuple[VideoMetadata, dict[str, Any]]:
    """Metadati completi via yt-dlp. Ritorna anche l'info dict grezzo."""
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise ExtractionError(
            "yt-dlp non installato. Esegui: pip install -r requirements.txt"
        ) from exc

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(canonical_url(video_id), download=False)
    except DownloadError as exc:
        message = str(exc).lower()
        if any(k in message for k in ("private", "unavailable", "removed", "age", "members-only")):
            raise VideoUnavailableError(
                f"Video non accessibile ({video_id}): privato, rimosso o con restrizioni. "
                f"Dettaglio: {exc}"
            ) from exc
        raise ExtractionError(f"yt-dlp non riesce a leggere il video {video_id}: {exc}") from exc
    except Exception as exc:  # rete, parsing, estrattore rotto...
        raise ExtractionError(f"Errore inatteso da yt-dlp su {video_id}: {exc}") from exc

    metadata = VideoMetadata(
        video_id=video_id,
        url=canonical_url(video_id),
        title=info.get("title") or "",
        channel=info.get("uploader") or info.get("channel") or "",
        channel_url=info.get("channel_url") or info.get("uploader_url") or "",
        duration_seconds=int(info.get("duration") or 0),
        upload_date=_normalize_upload_date(info.get("upload_date")),
        description=info.get("description") or "",
        tags=list(info.get("tags") or [])[:20],
        view_count=info.get("view_count"),
        thumbnail=info.get("thumbnail") or "",
        source="yt-dlp",
    )
    return metadata, info


def _metadata_from_oembed(video_id: str) -> VideoMetadata:
    """Fallback minimale: endpoint oEmbed pubblico (titolo + canale)."""
    url = f"https://www.youtube.com/oembed?url={canonical_url(video_id)}&format=json"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            raise VideoUnavailableError(
                f"Video {video_id} non accessibile pubblicamente (HTTP {exc.code})."
            ) from exc
        raise ExtractionError(f"oEmbed ha risposto HTTP {exc.code} per {video_id}.") from exc
    except Exception as exc:
        raise ExtractionError(f"Impossibile contattare oEmbed per {video_id}: {exc}") from exc

    return VideoMetadata(
        video_id=video_id,
        url=canonical_url(video_id),
        title=payload.get("title") or "",
        channel=payload.get("author_name") or "",
        channel_url=payload.get("author_url") or "",
        thumbnail=payload.get("thumbnail_url") or "",
        source="oembed",
    )


def fetch_metadata(video_id: str) -> tuple[VideoMetadata, dict[str, Any]]:
    """Metadati con fallback: yt-dlp -> oEmbed."""
    try:
        return _metadata_from_ytdlp(video_id)
    except VideoUnavailableError:
        raise  # inutile insistere: il video non e accessibile
    except ExtractionError as exc:
        log.warning("Metadati via yt-dlp falliti (%s); provo con oEmbed.", exc)
        return _metadata_from_oembed(video_id), {}


# ----------------------------------------------------------- trascrizione: API


def _segments_from_api(raw: Sequence[Any]) -> list[TranscriptSegment]:
    """Normalizza sia i dict (API 0.6.x) sia gli snippet oggetto (API 1.x)."""
    segments: list[TranscriptSegment] = []
    for item in raw:
        if isinstance(item, dict):
            start = item.get("start", 0)
            duration = item.get("duration", 0)
            text = item.get("text", "")
        else:
            start = getattr(item, "start", 0)
            duration = getattr(item, "duration", 0)
            text = getattr(item, "text", "")
        text = re.sub(r"\s+", " ", str(text)).strip()
        if text:
            segments.append(TranscriptSegment(float(start or 0), float(duration or 0), text))
    return segments


def _transcript_via_api(
    video_id: str, languages: Sequence[str]
) -> tuple[list[TranscriptSegment], str]:
    """youtube-transcript-api, compatibile con le versioni 0.6.x e 1.x."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise TranscriptUnavailableError(
            "youtube-transcript-api non installato. Esegui: pip install -r requirements.txt"
        ) from exc

    wanted = list(languages) or ["en"]

    # --- API 1.x: istanza + .fetch() / .list()
    if hasattr(YouTubeTranscriptApi, "fetch"):
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(video_id, languages=wanted)
            language = getattr(fetched, "language_code", wanted[0])
            return _segments_from_api(list(fetched)), language
        except Exception as exc:
            # ultimo tentativo: qualunque lingua disponibile
            try:
                transcript = next(iter(api.list(video_id)))
                return (
                    _segments_from_api(list(transcript.fetch())),
                    getattr(transcript, "language_code", ""),
                )
            except Exception:
                raise TranscriptUnavailableError(
                    f"youtube-transcript-api non trova sottotitoli per {video_id}: {exc}"
                ) from exc

    # --- API 0.6.x: metodi statici
    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id, languages=wanted)
        return _segments_from_api(raw), wanted[0]
    except Exception as exc:
        try:
            transcript = next(iter(YouTubeTranscriptApi.list_transcripts(video_id)))
            return (
                _segments_from_api(transcript.fetch()),
                getattr(transcript, "language_code", ""),
            )
        except Exception:
            raise TranscriptUnavailableError(
                f"youtube-transcript-api non trova sottotitoli per {video_id}: {exc}"
            ) from exc


# -------------------------------------------------------- trascrizione: yt-dlp

_VTT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
_VTT_TAGS = re.compile(r"<[^>]+>")


def _parse_json3(payload: str) -> list[TranscriptSegment]:
    """Parsa il formato `json3` delle caption YouTube."""
    data = json.loads(payload)
    segments: list[TranscriptSegment] = []
    for event in data.get("events") or []:
        text = "".join(seg.get("utf8", "") for seg in event.get("segs") or [])
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        start_ms = event.get("tStartMs", 0) or 0
        dur_ms = event.get("dDurationMs", 0) or 0
        segments.append(TranscriptSegment(start_ms / 1000.0, dur_ms / 1000.0, text))
    return segments


def _parse_vtt(payload: str) -> list[TranscriptSegment]:
    """Parsa WebVTT/SRT deduplicando le righe ripetute delle auto-caption."""
    segments: list[TranscriptSegment] = []
    window = {"start": 0.0, "end": 0.0}
    buffer: list[str] = []

    def flush() -> None:
        text = _VTT_TAGS.sub("", " ".join(buffer))
        text = re.sub(r"\s+", " ", text).strip()
        if text and (not segments or segments[-1].text != text):
            duration = max(0.0, window["end"] - window["start"])
            segments.append(TranscriptSegment(window["start"], duration, text))

    for raw_line in io.StringIO(payload):
        line = raw_line.strip()
        match = _VTT_TIME.search(line)
        if match:
            flush()
            buffer = []
            h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in match.groups())
            window["start"] = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            window["end"] = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
            continue
        if not line or line.isdigit():
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        buffer.append(line)
    flush()
    return segments


def _pick_caption_track(info: dict[str, Any], languages: Sequence[str]) -> tuple[str, str] | None:
    """Sceglie (url, lingua) dai sottotitoli di yt-dlp: manuali prima, poi auto."""
    preferred_ext = ("json3", "srv3", "vtt", "srt")
    for source_key in ("subtitles", "automatic_captions"):
        tracks = info.get(source_key) or {}
        if not tracks:
            continue
        # ordine: per ogni lingua richiesta prima il match esatto, poi quello
        # parziale (it -> it-IT); solo alla fine tutte le altre lingue presenti
        candidates: list[str] = []
        for wanted in languages:
            if wanted in tracks and wanted not in candidates:
                candidates.append(wanted)
            candidates += [
                lang for lang in tracks if lang.startswith(wanted) and lang not in candidates
            ]
        candidates += [lang for lang in tracks if lang not in candidates]
        for lang in candidates:
            formats = tracks.get(lang) or []
            for ext in preferred_ext:
                for fmt in formats:
                    if fmt.get("ext") == ext and fmt.get("url"):
                        return fmt["url"], lang
            if formats and formats[0].get("url"):
                return formats[0]["url"], lang
    return None


def _transcript_via_ytdlp(
    video_id: str, languages: Sequence[str], info: dict[str, Any] | None = None
) -> tuple[list[TranscriptSegment], str]:
    """Fallback: scarica direttamente la caption track esposta da yt-dlp."""
    if not info:
        _, info = _metadata_from_ytdlp(video_id)
    picked = _pick_caption_track(info, list(languages) or ["en"])
    if not picked:
        raise TranscriptUnavailableError(f"yt-dlp non espone caption track per {video_id}.")
    url, language = picked
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise TranscriptUnavailableError(f"Download della caption track fallito: {exc}") from exc

    segments = _parse_json3(payload) if payload.lstrip().startswith("{") else _parse_vtt(payload)
    if not segments:
        raise TranscriptUnavailableError("Caption track scaricata ma vuota dopo il parsing.")
    return segments, language


# -------------------------------------------------------- audio (fallback LLM)


def download_audio(video_id: str, out_dir: Path | None = None) -> Path:
    """Scarica la traccia audio (per la trascrizione con LLM multimodale).

    Richiede yt-dlp; ffmpeg e consigliato ma non obbligatorio: senza ffmpeg si
    tiene il formato nativo (webm/m4a), accettato dai modelli multimodali.
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise ExtractionError("yt-dlp non installato: impossibile scaricare l'audio.") from exc

    target_dir = Path(out_dir or Path(tempfile.mkdtemp(prefix="yt-learner-")))
    target_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": str(target_dir / f"{video_id}.%(ext)s"),
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(canonical_url(video_id), download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as exc:
        raise ExtractionError(f"Download audio fallito per {video_id}: {exc}") from exc

    if not path.exists():
        matches = sorted(target_dir.glob(f"{video_id}.*"))
        if not matches:
            raise ExtractionError("Download audio completato ma il file non e stato trovato.")
        path = matches[0]
    return path


# ----------------------------------------------------------------- entry point


def fetch_transcript(
    video_id: str,
    languages: Sequence[str] = ("it", "en"),
    info: dict[str, Any] | None = None,
) -> tuple[list[TranscriptSegment], str, str]:
    """Trascrizione con fallback a cascata (API -> yt-dlp).

    Ritorna `(segmenti, sorgente, lingua)`.
    """
    errors: list[str] = []
    try:
        segments, language = _transcript_via_api(video_id, languages)
        if segments:
            return segments, "youtube-transcript-api", language
        errors.append("youtube-transcript-api: risultato vuoto")
    except TranscriptUnavailableError as exc:
        errors.append(str(exc))
        log.info("Fallback su yt-dlp per la trascrizione: %s", exc)

    try:
        segments, language = _transcript_via_ytdlp(video_id, languages, info)
        return segments, "yt-dlp", language
    except ExtractionError as exc:
        errors.append(str(exc))

    raise TranscriptUnavailableError(
        "Nessuna trascrizione disponibile.\n  - " + "\n  - ".join(errors)
    )


def extract(
    url: str,
    languages: Sequence[str] = ("it", "en"),
    *,
    allow_audio_fallback: bool = False,
    audio_transcriber=None,
) -> VideoData:
    """Pipeline completa: url -> metadati -> trascrizione.

    `audio_transcriber` e una callable `(Path) -> list[TranscriptSegment]`
    (di norma `llm_processor.transcribe_audio`), usata solo se i sottotitoli non
    esistono e l'utente ha passato `--allow-audio-fallback`.
    """
    video_id = extract_video_id(url)
    metadata, info = fetch_metadata(video_id)

    try:
        segments, source, language = fetch_transcript(video_id, languages, info)
    except TranscriptUnavailableError:
        if not (allow_audio_fallback and audio_transcriber):
            raise
        log.warning("Nessun sottotitolo: scarico l'audio e trascrivo con LLM multimodale.")
        audio_path = download_audio(video_id)
        segments = audio_transcriber(audio_path)
        if not segments:
            raise TranscriptUnavailableError(
                "Anche la trascrizione audio con LLM non ha prodotto testo."
            )
        source = "llm-audio"
        language = languages[0] if languages else "en"

    return VideoData(
        metadata=metadata,
        segments=segments,
        transcript_source=source,
        transcript_language=language,
    )
