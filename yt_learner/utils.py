"""Helper condivisi: slug, timestamp, YAML frontmatter, IO su file."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------- slug / testo

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 60) -> str:
    """Trasforma un titolo in uno slug kebab-case sicuro per filesystem."""
    if not text:
        return "untitled"
    # rimuove accenti (è -> e) mantenendo i caratteri ascii
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_text).strip("-")
    if not slug:
        return "untitled"
    if len(slug) > max_len:
        # taglia sull'ultimo trattino per non spezzare una parola a metà
        slug = slug[:max_len].rsplit("-", 1)[0] or slug[:max_len]
    return slug.strip("-")


def fmt_timestamp(seconds: float | int | None) -> str:
    """Secondi -> `MM:SS` (o `HH:MM:SS` se il video supera l'ora)."""
    if seconds is None:
        return "00:00"
    total = int(max(0, float(seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timestamp(value: str) -> int:
    """`MM:SS` / `HH:MM:SS` -> secondi. Ritorna 0 se non parsabile."""
    if not value:
        return 0
    parts = [p for p in str(value).strip().strip("[]").split(":") if p != ""]
    try:
        numbers = [int(float(p)) for p in parts]
    except ValueError:
        return 0
    seconds = 0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def snippet(text: str, limit: int = 160) -> str:
    """Riassunto grezzo su una riga: spazi normalizzati e taglio su parola."""
    if not text:
        return ""
    single_line = re.sub(r"\s+", " ", str(text)).strip()
    if len(single_line) <= limit:
        return single_line
    cut = single_line[:limit].rsplit(" ", 1)[0] or single_line[:limit]
    return cut.rstrip(" ,;:.-") + "..."


def truncate(text: str, limit: int) -> str:
    """Taglia un testo lungo aggiungendo un marcatore esplicito di troncamento."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[... trascrizione troncata ...]"


# ------------------------------------------------------------------ YAML light

def _yaml_scalar(value: Any) -> str:
    """Serializza uno scalare in YAML, quotando quando necessario."""
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    needs_quotes = (
        text == ""
        or text[0] in "-?:,[]{}#&*!|>'\"%@`"
        or ": " in text
        or text.endswith(":")
        or "\n" in text
        or text.strip() != text
    )
    if needs_quotes:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        return f'"{escaped}"'
    return text


def yaml_frontmatter(data: dict[str, Any]) -> str:
    """Costruisce un blocco frontmatter YAML (scalari + liste di scalari)."""
    lines = ["---"]
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            items = [str(v) for v in value if str(v).strip()]
            if not items:
                continue
            lines.append(f"{key}: [{', '.join(_yaml_scalar(i) for i in items)}]")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {_yaml_scalar(sub_value)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    return "\n".join(lines)


# -------------------------------------------------------------------------- IO

def write_text(path: Path, content: str, *, overwrite: bool = True) -> Path:
    """Scrive un file di testo creando le cartelle mancanti.

    Se `overwrite` è False e il file esiste, aggiunge un suffisso numerico.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        stem, suffix = path.stem, path.suffix
        counter = 2
        while path.exists():
            path = path.with_name(f"{stem}-{counter}{suffix}")
            counter += 1
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def dump_json(path: Path, data: Any) -> Path:
    """Salva un oggetto in JSON indentato (usato per l'output `--dump-json`)."""
    return write_text(Path(path), json.dumps(data, ensure_ascii=False, indent=2))


def bullet_list(items: Iterable[Any], prefix: str = "- ") -> str:
    """Lista markdown da un iterabile, stringa vuota se non c'è nulla."""
    lines = [f"{prefix}{str(item).strip()}" for item in items if str(item).strip()]
    return "\n".join(lines)
