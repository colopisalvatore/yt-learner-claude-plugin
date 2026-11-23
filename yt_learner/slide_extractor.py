"""Estrazione delle slide reali del video (screenshot dei frame).

Due modalita:
  - `timeline` (default): uno screenshot per ogni voce della timeline generata,
    preso qualche secondo dopo il timestamp per saltare le transizioni;
  - `scene`: rilevamento automatico dei cambi di inquadratura con ffmpeg, utile
    quando il video e una sequenza di slide e vuoi prenderle tutte.

In entrambi i casi i frame quasi identici vengono scartati con un dHash
calcolato su una miniatura 9x8 in scala di grigi (nessuna dipendenza extra:
la miniatura la produce ffmpeg stesso).

Richiede `ffmpeg` nel PATH e `yt-dlp` per scaricare il video.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .utils import fmt_timestamp, slugify
from .yt_extractor import ExtractionError, canonical_url

log = logging.getLogger(__name__)

# distanza di Hamming sotto la quale due frame sono considerati la stessa slide
DEFAULT_HASH_DISTANCE = 6


class SlideError(ExtractionError):
    """Errore durante il download del video o l'estrazione dei frame."""


@dataclass
class Slide:
    """Uno screenshot estratto dal video."""

    seconds: int
    path: Path
    label: str = ""

    @property
    def timestamp(self) -> str:
        return fmt_timestamp(self.seconds)


# --------------------------------------------------------------- dipendenze


def ffmpeg_path() -> str:
    """Percorso di ffmpeg, o errore parlante se manca."""
    found = shutil.which("ffmpeg")
    if not found:
        raise SlideError(
            "ffmpeg non trovato nel PATH: serve per estrarre le slide. "
            "Installalo (winget install Gyan.FFmpeg / apt install ffmpeg) "
            "oppure lancia senza --slides."
        )
    return found


def _run(cmd: Sequence[str], *, capture_stderr: bool = False) -> subprocess.CompletedProcess:
    """Esegue un comando esterno senza aprire finestre e senza bloccarsi su stdin."""
    return subprocess.run(
        list(cmd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
        check=False,
    )


# ------------------------------------------------------------------ download


def download_video(video_id: str, out_dir: Path | None = None, max_height: int = 720) -> Path:
    """Scarica la sola traccia video (niente audio: servono solo i frame)."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise SlideError("yt-dlp non installato: impossibile scaricare il video.") from exc

    target_dir = Path(out_dir or Path(tempfile.mkdtemp(prefix="yt-learner-video-")))
    target_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,   # niente barra di avanzamento nel log della CLI
        "noplaylist": True,
        "format": f"bv*[height<={max_height}]/b[height<={max_height}]/best",
        "outtmpl": str(target_dir / f"{video_id}.%(ext)s"),
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(canonical_url(video_id), download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as exc:
        raise SlideError(f"Download del video {video_id} fallito: {exc}") from exc

    if not path.exists():
        matches = sorted(target_dir.glob(f"{video_id}.*"))
        if not matches:
            raise SlideError("Video scaricato ma il file non e stato trovato.")
        path = matches[0]
    return path


# ------------------------------------------------------------------- hashing


def _thumbnail_bytes(video_path: Path, seconds: float) -> bytes:
    """Miniatura 9x8 in scala di grigi del frame indicato (72 byte)."""
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, seconds):.2f}", "-i", str(video_path),
        "-frames:v", "1", "-vf", "scale=9:8,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    return _run(cmd).stdout


def frame_hash(video_path: Path, seconds: float) -> int | None:
    """dHash a 64 bit del frame al secondo indicato (None se non estraibile)."""
    data = _thumbnail_bytes(video_path, seconds)
    if len(data) < 72:
        return None
    bits = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            bits = (bits << 1) | (1 if data[base + col] < data[base + col + 1] else 0)
    return bits


def frame_detail(video_path: Path, seconds: float) -> float:
    """Quanto "denso" e il frame: deviazione standard della luminanza.

    Serve a scartare i frame vuoti (b-roll, schermate bianche, stacchi) e a
    scegliere, dentro una finestra, quello che mostra davvero la slide.
    """
    data = _thumbnail_bytes(video_path, seconds)
    if len(data) < 72:
        return 0.0
    pixels = list(data[:72])
    mean = sum(pixels) / len(pixels)
    variance = sum((value - mean) ** 2 for value in pixels) / len(pixels)
    return variance ** 0.5


def _hamming(first: int, second: int) -> int:
    return bin(first ^ second).count("1")


def _is_duplicate(candidate: int | None, seen: Iterable[int], threshold: int) -> bool:
    if candidate is None:
        return False
    return any(_hamming(candidate, other) <= threshold for other in seen)


# ------------------------------------------------------------------- estrazione


def grab_frame(video_path: Path, seconds: float, out_path: Path, width: int = 1280) -> bool:
    """Salva un singolo frame come JPEG. Ritorna False se ffmpeg non produce nulla."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, seconds):.2f}", "-i", str(video_path),
        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3",
        str(out_path),
    ]
    _run(cmd)
    return out_path.exists() and out_path.stat().st_size > 0


def _best_candidate(
    video_path: Path, seconds: float, offset: float, probes: int, spread: float
) -> tuple[float, float]:
    """Dentro una finestra dopo il timestamp, sceglie il frame piu informativo.

    Evita di catturare la transizione, il b-roll o lo stacco: campiona qualche
    istante e tiene quello con piu dettaglio. Ritorna `(secondo, dettaglio)`.
    """
    best_at, best_detail = seconds + offset, -1.0
    for index in range(max(1, probes)):
        at = seconds + offset + index * spread
        detail = frame_detail(video_path, at)
        if detail > best_detail:
            best_at, best_detail = at, detail
    return best_at, best_detail


def extract_at_timestamps(
    video_path: Path,
    entries: Sequence[dict],
    out_dir: Path,
    *,
    offset: float = 3.0,
    width: int = 1280,
    dedup_distance: int = DEFAULT_HASH_DISTANCE,
    probes: int = 3,
    probe_spread: float = 5.0,
    min_detail: float = 6.0,
) -> list[Slide]:
    """Uno screenshot per voce di timeline (`seconds` + `label`).

    `offset` sposta la cattura in avanti per non prendere la transizione;
    `probes` campioni successivi permettono di scegliere il frame migliore.
    """
    slides: list[Slide] = []
    hashes: list[int] = []
    for entry in entries:
        seconds = float(entry.get("seconds") or 0)
        at, detail = _best_candidate(video_path, seconds, offset, probes, probe_spread)
        if detail < min_detail:
            log.info(
                "Slide a %s scartata: frame quasi vuoto (dettaglio %.1f).",
                fmt_timestamp(seconds),
                detail,
            )
            continue
        digest = frame_hash(video_path, at)
        if _is_duplicate(digest, hashes, dedup_distance):
            log.info("Slide a %s scartata: identica a una precedente.", fmt_timestamp(seconds))
            continue
        name = f"{int(seconds):05d}-{slugify(entry.get('label') or 'slide', 40)}.jpg"
        target = out_dir / name
        if not grab_frame(video_path, at, target, width=width):
            log.warning("Frame a %s non estratto.", fmt_timestamp(seconds))
            continue
        if digest is not None:
            hashes.append(digest)
        slides.append(Slide(seconds=int(seconds), path=target, label=entry.get("label") or ""))
    return slides


def _thumbnail_stream(
    video_path: Path, interval: float, cols: int = 17, rows: int = 16
) -> list[tuple[float, bytes]]:
    """Miniature grigie a intervallo fisso, in UNA sola passata di ffmpeg.

    Molto piu veloce di un seek per campione: il video viene decodificato una
    volta sola e ffmpeg emette direttamente le miniature.
    """
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps=1/{max(0.5, interval)},scale={cols}:{rows},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    data = _run(cmd).stdout
    size = cols * rows
    return [
        (index * interval, data[index * size : (index + 1) * size])
        for index in range(len(data) // size)
    ]


def _dhash(thumb: bytes, cols: int = 17, rows: int = 16) -> int:
    """dHash su miniatura arbitraria: (cols-1) x rows bit."""
    bits = 0
    for row in range(rows):
        base = row * cols
        for col in range(cols - 1):
            bits = (bits << 1) | (1 if thumb[base + col] < thumb[base + col + 1] else 0)
    return bits


def _thumb_detail(thumb: bytes) -> float:
    if not thumb:
        return 0.0
    mean = sum(thumb) / len(thumb)
    return (sum((value - mean) ** 2 for value in thumb) / len(thumb)) ** 0.5


def extract_by_sampling(
    video_path: Path,
    out_dir: Path,
    *,
    interval: float = 5.0,
    # misurato su tutorial con webcam in sovrimpressione: il movimento del volto
    # sta sotto i ~33 bit su 256, un cambio slide vero supera i 58
    change_bits: int = 45,
    min_gap: float = 6.0,
    width: int = 1280,
    max_slides: int = 80,
    min_detail: float = 6.0,
    grab_offset: float = 0.8,
) -> list[Slide]:
    """Campiona il video e tiene un frame ogni volta che l'immagine CAMBIA.

    Pensata per i deck: quando le slide condividono lo stesso template, il
    rilevamento scena di ffmpeg non scatta (cambia solo il testo), mentre il
    confronto tra hash successivi vede la differenza.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = _thumbnail_stream(video_path, interval)
    if not samples:
        raise SlideError("ffmpeg non ha prodotto campioni: video illeggibile?")

    slides: list[Slide] = []
    previous_hash: int | None = None
    last_kept = -min_gap
    for seconds, thumb in samples:
        if len(thumb) < 2 or _thumb_detail(thumb) < min_detail:
            continue
        digest = _dhash(thumb)
        changed = previous_hash is None or _hamming(digest, previous_hash) >= change_bits
        previous_hash = digest
        if not changed or seconds - last_kept < min_gap:
            continue
        target = out_dir / f"{int(seconds):05d}-slide.jpg"
        if not grab_frame(video_path, seconds + grab_offset, target, width=width):
            continue
        last_kept = seconds
        slides.append(Slide(seconds=int(seconds), path=target))
        if len(slides) >= max_slides:
            log.warning("Raggiunto il limite di %s slide: le successive sono ignorate.", max_slides)
            break
    return slides


_PTS_TIME = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


def extract_scene_changes(
    video_path: Path,
    out_dir: Path,
    *,
    threshold: float = 0.25,
    min_gap: float = 4.0,
    width: int = 1280,
    max_slides: int = 60,
    dedup_distance: int = DEFAULT_HASH_DISTANCE,
) -> list[Slide]:
    """Screenshot a ogni cambio scena rilevato da ffmpeg (slide che cambiano)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = out_dir / "_raw"
    workspace.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(), "-hide_banner", "-y", "-i", str(video_path),
        "-vf", f"select='gt(scene,{threshold})',showinfo,scale={width}:-2",
        "-vsync", "vfr", "-q:v", "3",
        str(workspace / "scene-%04d.jpg"),
    ]
    result = _run(cmd, capture_stderr=True)
    stderr = (result.stderr or b"").decode("utf-8", errors="replace")
    times = [float(match) for match in _PTS_TIME.findall(stderr)]
    frames = sorted(workspace.glob("scene-*.jpg"))
    if not frames:
        raise SlideError(
            "Nessun cambio scena rilevato: prova ad abbassare --slides-threshold "
            "oppure usa --slides-mode timeline."
        )

    slides: list[Slide] = []
    hashes: list[int] = []
    last_kept = -min_gap
    for index, frame in enumerate(frames):
        seconds = times[index] if index < len(times) else float(index)
        if seconds - last_kept < min_gap:
            frame.unlink(missing_ok=True)
            continue
        digest = frame_hash(video_path, seconds + 0.5)
        if _is_duplicate(digest, hashes, dedup_distance):
            frame.unlink(missing_ok=True)
            continue
        target = out_dir / f"{int(seconds):05d}-slide.jpg"
        frame.replace(target)
        if digest is not None:
            hashes.append(digest)
        last_kept = seconds
        slides.append(Slide(seconds=int(seconds), path=target))
        if len(slides) >= max_slides:
            log.warning("Raggiunto il limite di %s slide: le successive sono ignorate.", max_slides)
            break

    for leftover in workspace.glob("*"):
        leftover.unlink(missing_ok=True)
    workspace.rmdir()
    return slides


# ---------------------------------------------------------------- entry point


def extract_slides(
    video_id: str,
    out_dir: Path,
    *,
    mode: str = "timeline",
    timeline: Sequence[dict] = (),
    max_height: int = 720,
    width: int = 1280,
    offset: float = 3.0,
    threshold: float = 0.25,
    min_gap: float = 4.0,
    max_slides: int = 60,
    interval: float = 5.0,
    change_bits: int = 45,
    keep_video: Path | None = None,
    video_path: Path | None = None,
) -> list[Slide]:
    """Scarica il video (se serve) ed estrae le slide nella cartella indicata."""
    ffmpeg_path()  # fallisce subito e con un messaggio chiaro se manca
    out_dir = Path(out_dir)
    downloaded = None
    if video_path is None:
        video_path = downloaded = download_video(
            video_id, keep_video or None, max_height=max_height
        )
    try:
        if mode == "sample":
            return extract_by_sampling(
                video_path,
                out_dir,
                interval=interval,
                change_bits=change_bits,
                min_gap=max(min_gap, interval),
                width=width,
                max_slides=max_slides,
            )
        if mode == "scene":
            return extract_scene_changes(
                video_path,
                out_dir,
                threshold=threshold,
                min_gap=min_gap,
                width=width,
                max_slides=max_slides,
            )
        if not timeline:
            raise SlideError(
                "Modalita timeline senza voci di timeline: genera prima la scheda "
                "knowledge oppure usa --slides-mode scene."
            )
        return extract_at_timestamps(
            video_path, list(timeline), out_dir, offset=offset, width=width
        )
    finally:
        # il video temporaneo si butta, a meno che l'utente non lo voglia tenere
        if downloaded and not keep_video:
            try:
                downloaded.unlink(missing_ok=True)
                parent = downloaded.parent
                if parent.name.startswith("yt-learner-video-") and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError as exc:
                log.debug("Pulizia del video temporaneo fallita: %s", exc)
