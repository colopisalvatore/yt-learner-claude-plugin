"""Rende la spec JSON della skill in una cartella `skills/<skill-name>/`.

Struttura prodotta:

    skills/<skill-name>/
      SKILL.md          <- frontmatter + workflow + snippet + validazione
      scripts/...       <- solo se la spec contiene asset eseguibili
      examples/...      <- solo se la spec contiene template/esempi
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .utils import slugify, write_text, yaml_frontmatter
from .yt_extractor import VideoData

log = logging.getLogger(__name__)

# Estensioni ammesse per gli asset: evita che il modello scriva dove non deve.
ALLOWED_ASSET_SUFFIXES = {
    ".py", ".js", ".ts", ".sh", ".bash", ".ps1", ".sql", ".json", ".yaml", ".yml",
    ".toml", ".env", ".md", ".txt", ".ini", ".cfg", ".html", ".css", ".dockerfile",
}


def find_skill_by_source(skills_dir: Path, source_url: str) -> str | None:
    """Nome della skill gia generata da questo video, se esiste.

    Il modello propone un nome diverso a ogni giro: senza questo controllo
    rieseguire sullo stesso video creerebbe una seconda cartella invece di
    aggiornare la prima. Il legame e `source_url` nel frontmatter.
    """
    skills_dir = Path(skills_dir)
    if not source_url or not skills_dir.is_dir():
        return None
    needle = f"source_url: {source_url}"
    for candidate in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            head = candidate.read_text(encoding="utf-8", errors="replace")[:1500]
        except OSError:
            continue
        if needle in head:
            return candidate.parent.name
    return None


@dataclass
class SkillResult:
    """Esito della scrittura di una skill."""

    skill_name: str
    skill_dir: Path
    skill_file: Path
    assets: list[Path] = field(default_factory=list)


# ------------------------------------------------------------------- rendering


def _render_prerequisites(prerequisites: dict[str, Any]) -> str:
    blocks: list[str] = []
    labels = {
        "tools": "**Tool / CLI**",
        "libraries": "**Librerie**",
        "credentials": "**Credenziali & variabili d'ambiente**",
        "assumptions": "**Assunzioni**",
    }
    for key, label in labels.items():
        items = prerequisites.get(key) or []
        if not items:
            continue
        lines = "\n".join(f"- {item}" for item in items)
        blocks.append(f"{label}\n{lines}")
    return "\n\n".join(blocks) or "_Nessun prerequisito esplicito nel video._"


def _render_workflow(workflow: list[dict[str, Any]]) -> str:
    if not workflow:
        return "_Il video non descrive una procedura sequenziale._"
    blocks: list[str] = []
    for step in workflow:
        header = f"### Passo {step['step']} — {step['title']}"
        if step.get("timestamp"):
            header += f"  `[{step['timestamp']}]`"
        parts = [header]
        if step.get("instructions"):
            parts.append(step["instructions"])
        commands = step.get("commands") or []
        if commands:
            body = "\n".join(commands)
            parts.append(f"```bash\n{body}\n```")
        if step.get("expected_result"):
            parts.append(f"**Risultato atteso:** {step['expected_result']}")
        blocks.append("\n\n".join(parts))
    return "\n\n".join(blocks)


def _render_snippets(snippets: list[dict[str, Any]]) -> str:
    if not snippets:
        return "_Nessuno snippet estratto dal video._"
    blocks: list[str] = []
    for snippet in snippets:
        title = snippet.get("title") or "Snippet"
        header = f"### {title}"
        if snippet.get("filename"):
            header += f" — `{snippet['filename']}`"
        parts = [header]
        if snippet.get("explanation"):
            parts.append(snippet["explanation"])
        language = snippet.get("language") or "text"
        parts.append(f"```{language}\n{snippet.get('code', '').rstrip()}\n```")
        blocks.append("\n\n".join(parts))
    return "\n\n".join(blocks)


def _render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], empty: str) -> str:
    """Tabella markdown; `columns` e una lista di (chiave, intestazione)."""
    if not rows:
        return f"_{empty}_"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = str(row.get(key, "") or "").replace("|", "\\|").replace("\n", " ")
            cells.append(value or "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_assets(assets: list[dict[str, Any]]) -> str:
    if not assets:
        return ""
    lines = [f"- `{asset['path']}` — {asset.get('purpose') or 'file di supporto'}" for asset in assets]
    return "## Asset inclusi\n\n" + "\n".join(lines)


def render_skill_markdown(
    spec: dict[str, Any],
    video: VideoData,
    *,
    knowledge_rel_path: str | None = None,
    learned_date: str | None = None,
) -> str:
    """Compone il testo completo di `SKILL.md`."""
    meta = video.metadata
    frontmatter = yaml_frontmatter(
        {
            "name": spec["skill_name"],
            "description": spec["description"],
            "version": spec.get("version") or "0.1.0",
            "tags": spec.get("tags") or [],
            "triggers": spec.get("triggers") or [],
            "source_video": meta.title,
            "source_url": meta.url,
            "source_channel": meta.channel,
            "generated_by": "yt-learner",
            "generated_at": learned_date or date.today().isoformat(),
            "related_knowledge": knowledge_rel_path or "",
        }
    )

    sections = [
        frontmatter,
        f"# {spec['skill_name']}",
        f"> {spec['description']}",
    ]

    if spec.get("objective"):
        sections.append(f"**Obiettivo:** {spec['objective']}")

    sections.append(
        "## Prerequisiti & dipendenze\n\n" + _render_prerequisites(spec.get("prerequisites") or {})
    )
    sections.append("## Workflow passo-passo\n\n" + _render_workflow(spec.get("workflow") or []))
    sections.append(
        "## Snippet di codice & comandi\n\n" + _render_snippets(spec.get("code_snippets") or [])
    )
    sections.append(
        "## Error handling & troubleshooting\n\n"
        + _render_table(
            spec.get("error_handling") or [],
            [("error", "Errore / sintomo"), ("cause", "Causa"), ("fix", "Soluzione")],
            "Nessun errore comune citato nel video.",
        )
    )
    sections.append(
        "## Criteri di validazione\n\n"
        + _render_table(
            spec.get("validation") or [],
            [("check", "Cosa verificare"), ("how", "Come"), ("success_criteria", "Criterio di successo")],
            "Nessun criterio di validazione esplicito: verificare a mano l'output finale.",
        )
    )

    assets_block = _render_assets(spec.get("assets") or [])
    if assets_block:
        sections.append(assets_block)

    references = spec.get("references") or []
    reference_lines = [f"- {ref}" for ref in references]
    reference_lines.append(f"- Video sorgente: [{meta.title or meta.url}]({meta.url})")
    if knowledge_rel_path:
        reference_lines.append(f"- Scheda knowledge nel brain: `{knowledge_rel_path}`")
    sections.append("## Riferimenti\n\n" + "\n".join(reference_lines))

    return "\n\n".join(section.strip() for section in sections if section.strip())


# -------------------------------------------------------------------- scrittura


def _safe_asset_path(skill_dir: Path, raw_path: str) -> Path | None:
    """Impedisce path traversal e estensioni non previste negli asset."""
    candidate = (skill_dir / raw_path).resolve()
    try:
        candidate.relative_to(skill_dir.resolve())
    except ValueError:
        log.warning("Asset scartato (path fuori dalla skill): %s", raw_path)
        return None
    suffix = candidate.suffix.lower()
    if suffix and suffix not in ALLOWED_ASSET_SUFFIXES:
        log.warning("Asset scartato (estensione non ammessa %s): %s", suffix, raw_path)
        return None
    return candidate


def build_skill(
    spec: dict[str, Any],
    video: VideoData,
    skills_dir: Path,
    *,
    skill_name: str | None = None,
    knowledge_rel_path: str | None = None,
    learned_date: str | None = None,
    overwrite: bool = True,
) -> SkillResult:
    """Scrive `SKILL.md` (+ eventuali asset) e ritorna i percorsi creati."""
    spec = dict(spec)
    if skill_name:
        spec["skill_name"] = slugify(skill_name)
    spec["skill_name"] = slugify(spec.get("skill_name") or video.metadata.title or "skill")

    skills_dir = Path(skills_dir)
    skill_dir = skills_dir / spec["skill_name"]
    if skill_dir.exists() and not overwrite:
        counter = 2
        while (skills_dir / f"{spec['skill_name']}-{counter}").exists():
            counter += 1
        spec["skill_name"] = f"{spec['skill_name']}-{counter}"
        skill_dir = skills_dir / spec["skill_name"]

    markdown = render_skill_markdown(
        spec, video, knowledge_rel_path=knowledge_rel_path, learned_date=learned_date
    )
    skill_file = write_text(skill_dir / "SKILL.md", markdown)

    written_assets: list[Path] = []
    for asset in spec.get("assets") or []:
        target = _safe_asset_path(skill_dir, asset["path"])
        if target is None:
            continue
        written_assets.append(write_text(target, asset["content"]))

    return SkillResult(
        skill_name=spec["skill_name"],
        skill_dir=skill_dir,
        skill_file=skill_file,
        assets=written_assets,
    )
