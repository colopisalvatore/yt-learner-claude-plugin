"""Rende la spec JSON di knowledge in una scheda markdown per il BRAIN.

Output: `<brain-dir>/<slug-video>.md`, con frontmatter compatibile sia con la
specifica richiesta (`title`, `source_url`, `channel`, `duration`,
`learned_date`, `related_skill`) sia con le convenzioni di un vault Manent
(`name`, `description`, `type`).

Opzionalmente aggiorna un indice `MEMORY.md` con un puntatore di una riga.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .utils import bullet_list, fmt_timestamp, slugify, snippet, write_text, yaml_frontmatter
from .yt_extractor import VideoData

log = logging.getLogger(__name__)


@dataclass
class KnowledgeResult:
    """Esito della scrittura della scheda di knowledge."""

    slug: str
    path: Path
    index_updated: bool = False


def find_note_by_source(brain_dir: Path, source_url: str) -> Path | None:
    """Scheda gia scritta per questo video, se c'e.

    Serve all'idempotenza: rieseguire sullo stesso video deve AGGIORNARE la
    scheda, non affiancarne una seconda con uno slug diverso (o con la data di
    oggi). Il legame e `source_url` nel frontmatter.
    """
    brain_dir = Path(brain_dir)
    if not source_url or not brain_dir.is_dir():
        return None
    needle = f"source_url: {source_url}"
    for candidate in sorted(brain_dir.glob("*.md")):
        try:
            head = candidate.read_text(encoding="utf-8", errors="replace")[:1500]
        except OSError:
            continue
        if needle in head:
            return candidate
    return None


def deep_link(video_id: str, seconds: int) -> str:
    """Link YouTube che parte dal timestamp indicato."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(max(0, seconds))}s"


# ------------------------------------------------------------------- rendering


def _render_concepts(concepts: list[dict[str, Any]]) -> str:
    if not concepts:
        return "_Nessun concetto teorico estratto._"
    blocks = []
    for item in concepts:
        header = f"### {item.get('concept') or 'Concetto'}"
        if item.get("timestamp"):
            header += f"  `[{item['timestamp']}]`"
        explanation = item.get("explanation") or ""
        blocks.append(f"{header}\n\n{explanation}".rstrip())
    return "\n\n".join(blocks)


def _render_timeline(
    timeline: list[dict[str, Any]],
    video_id: str,
    slides_by_second: dict[int, dict[str, Any]] | None = None,
) -> str:
    if not timeline:
        return "_Timeline non disponibile (trascrizione senza timestamp)._"
    slides_by_second = slides_by_second or {}
    lines = []
    for entry in timeline:
        timestamp = entry.get("timestamp") or "00:00"
        seconds = int(entry.get("seconds") or 0)
        link = deep_link(video_id, seconds)
        label = entry.get("label") or "Segmento"
        line = f"- [`[{timestamp}]`]({link}) — **{label}**"
        detail = (entry.get("detail") or "").strip()
        if detail:
            line += f": {detail}"
        lines.append(line)
        slide = slides_by_second.get(seconds)
        if slide:
            # la slide del video, agganciata al punto della timeline che la mostra
            lines.append(f"  <br>![slide {timestamp}]({slide['path']})")
    return "\n".join(lines)


def _render_slide_gallery(slides: list[dict[str, Any]], video_id: str) -> str:
    """Slide non agganciate a una voce di timeline: galleria in fondo."""
    if not slides:
        return ""
    blocks = ["## Slide estratte dal video", ""]
    for slide in slides:
        seconds = int(slide.get("seconds") or 0)
        timestamp = fmt_timestamp(seconds)
        caption = slide.get("label") or "slide"
        blocks.append(f"### [`[{timestamp}]`]({deep_link(video_id, seconds)}) {caption}".rstrip())
        blocks.append("")
        blocks.append(f"![slide {timestamp}]({slide['path']})")
        blocks.append("")
    return "\n".join(blocks).rstrip()


def _match_slides_to_timeline(
    timeline: list[dict[str, Any]],
    slides: Sequence[dict[str, Any]],
    window: int = 120,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Aggancia a ogni voce di timeline la prima slide che la illustra.

    La slide giusta e quella che compare subito DOPO il timestamp della voce
    (entro `window` secondi) e prima della voce successiva. Le altre restano
    per la galleria: cosi anche in modalita `sample` la timeline e illustrata
    senza perdere le slide in piu.
    """
    if not timeline or not slides:
        return {}, list(slides)

    ordered = sorted(slides, key=lambda item: int(item.get("seconds") or 0))
    entries = sorted(timeline, key=lambda item: int(item.get("seconds") or 0))
    matched: dict[int, dict[str, Any]] = {}
    used: set[int] = set()

    for index, entry in enumerate(entries):
        start = int(entry.get("seconds") or 0)
        next_start = (
            int(entries[index + 1].get("seconds") or 0) if index + 1 < len(entries) else None
        )
        limit = min(start + window, next_start) if next_start else start + window
        for position, slide in enumerate(ordered):
            if position in used:
                continue
            seconds = int(slide.get("seconds") or 0)
            if seconds < start:
                continue
            if seconds > limit:
                break
            matched[start] = slide
            used.add(position)
            break

    leftover = [slide for position, slide in enumerate(ordered) if position not in used]
    return matched, leftover


def render_knowledge_markdown(
    spec: dict[str, Any],
    video: VideoData,
    *,
    skill_rel_path: str | None = None,
    skill_name: str | None = None,
    learned_date: str | None = None,
    note_type: str = "reference",
    legacy_metadata_type: bool = True,
    note_name: str | None = None,
    slides: Sequence[dict[str, Any]] = (),
) -> str:
    """Compone il testo completo della scheda di knowledge."""
    meta = video.metadata
    learned = learned_date or date.today().isoformat()
    description = snippet(spec.get("one_liner") or spec.get("executive_summary") or "", 180)

    timeline = spec.get("timeline") or []
    slides_by_second, leftover_slides = _match_slides_to_timeline(timeline, slides)

    frontmatter = yaml_frontmatter(
        {
            "name": note_name or spec["slug"],
            "title": spec["title"],
            "description": description,
            "type": note_type,
            "source_url": meta.url,
            "channel": meta.channel,
            "duration": meta.duration_human,
            "published": meta.upload_date,
            "learned_date": learned,
            "related_skill": skill_rel_path or "",
            "tags": spec.get("tags") or [],
            "slides": len(slides) or None,
            "metadata": {"type": note_type} if legacy_metadata_type else None,
        }
    )

    sections = [frontmatter, f"# {spec['title']}"]
    if spec.get("one_liner"):
        sections.append(f"> {spec['one_liner']}")

    sections.append(
        "## Executive summary\n\n"
        + (spec.get("executive_summary") or "_Sintesi non disponibile._")
    )
    if spec.get("relevance"):
        sections.append("## Perche e rilevante\n\n" + spec["relevance"])

    sections.append("## Concetti chiave & teoria\n\n" + _render_concepts(spec.get("key_concepts") or []))

    if spec.get("architecture"):
        sections.append("## Architettura / flusso\n\n" + spec["architecture"])

    if spec.get("tools_mentioned"):
        sections.append("## Tool e librerie citati\n\n" + bullet_list(spec["tools_mentioned"]))

    sections.append(
        "## Timeline / segnalibri\n\n"
        + _render_timeline(timeline, meta.video_id, slides_by_second)
    )

    gallery = _render_slide_gallery(leftover_slides, meta.video_id)
    if gallery:
        sections.append(gallery)

    if spec.get("takeaways"):
        sections.append("## Takeaway operativi\n\n" + bullet_list(spec["takeaways"]))
    if spec.get("pitfalls"):
        sections.append("## Trappole & errori noti\n\n" + bullet_list(spec["pitfalls"]))
    if spec.get("open_questions"):
        sections.append("## Domande aperte\n\n" + bullet_list(spec["open_questions"]))

    cross_lines = [f"- Video sorgente: [{meta.title or meta.url}]({meta.url})"]
    if meta.channel:
        channel = f"[{meta.channel}]({meta.channel_url})" if meta.channel_url else meta.channel
        cross_lines.append(f"- Canale: {channel}")
    if skill_rel_path:
        cross_lines.append(f"- Skill eseguibile generata: `{skill_rel_path}`")
    if skill_name:
        cross_lines.append(f"- Wikilink skill: [[{skill_name}]]")
    cross_lines.append(
        f"- Trascrizione: fonte `{video.transcript_source}`, lingua "
        f"`{video.transcript_language or 'n/d'}`, {video.word_count} parole"
    )
    sections.append("## Cross-reference\n\n" + "\n".join(cross_lines))

    return "\n\n".join(section.strip() for section in sections if section.strip())


# -------------------------------------------------------------------- indice


def update_index(index_path: Path, slug: str, title: str, hook: str, note_rel_path: str) -> bool:
    """Aggiunge (o aggiorna) la riga-puntatore nell'indice del brain."""
    index_path = Path(index_path)
    # l'indice e a una riga per nota: niente newline nel titolo o nell'hook
    line = f"- [{snippet(title, 90)}]({note_rel_path}) — {snippet(hook, 140)}".rstrip(" —")
    try:
        existing = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    except OSError as exc:
        log.warning("Indice non leggibile (%s): salto l'aggiornamento.", exc)
        return False

    if f"({note_rel_path})" in existing:
        pattern = re.compile(rf"^.*\({re.escape(note_rel_path)}\).*$", re.MULTILINE)
        updated = pattern.sub(line, existing, count=1)
    else:
        header = "" if existing else "# MEMORY\n\nIndice delle schede di knowledge.\n"
        updated = f"{header}{existing.rstrip()}\n{line}\n" if existing else f"{header}\n{line}\n"

    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(updated.lstrip("\n"), encoding="utf-8")
    except OSError as exc:
        log.warning("Scrittura dell'indice fallita (%s).", exc)
        return False
    return True


# ------------------------------------------------------------------ scrittura


def write_knowledge(
    spec: dict[str, Any],
    video: VideoData,
    brain_dir: Path,
    *,
    skill_rel_path: str | None = None,
    skill_name: str | None = None,
    learned_date: str | None = None,
    overwrite: bool = True,
    index_path: Path | None = None,
    filename: str | None = None,
    note_type: str = "reference",
    legacy_metadata_type: bool = True,
    slides: Sequence[dict[str, Any]] = (),
) -> KnowledgeResult:
    """Scrive la scheda nel brain e, se richiesto, aggiorna l'indice."""
    spec = dict(spec)
    spec["slug"] = slugify(spec.get("slug") or spec.get("title") or video.metadata.title or video.metadata.video_id)
    target_name = filename or f"{spec['slug']}.md"
    # in un vault Manent il `name` canonico e lo slug del file, data inclusa
    note_name = Path(target_name).stem

    markdown = render_knowledge_markdown(
        spec,
        video,
        skill_rel_path=skill_rel_path,
        skill_name=skill_name,
        learned_date=learned_date,
        note_type=note_type,
        legacy_metadata_type=legacy_metadata_type,
        note_name=note_name,
        slides=slides,
    )
    brain_dir = Path(brain_dir)
    path = write_text(brain_dir / target_name, markdown, overwrite=overwrite)

    index_updated = False
    if index_path:
        hook = snippet(spec.get("one_liner") or spec.get("executive_summary") or "", 140)
        try:
            relative = path.relative_to(Path(index_path).parent).as_posix()
        except ValueError:
            relative = path.as_posix()
        index_updated = update_index(Path(index_path), spec["slug"], spec["title"], hook, relative)

    return KnowledgeResult(slug=spec["slug"], path=path, index_updated=index_updated)
