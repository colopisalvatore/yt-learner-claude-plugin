"""Destinazioni del brain ("preset"): dove e come scrivere la scheda.

Un preset descrive un vault: cartella, layout dei nomi file, `type` da mettere
nel frontmatter, indice da aggiornare e comando di lint da lanciare dopo la
scrittura. Cosi `--brain manent` scrive in un vault Manent con le sue convenzioni,
senza doversi ricordare cinque flag ogni volta.

Ordine di risoluzione:
  1. preset built-in (`default`, `manent`)
  2. `./yt-learner.json` nella cwd
  3. `~/.yt-learner.json`
  4. variabili d'ambiente (`MANENT_VAULT`, `MANENT_CLI`)

`--brain-dir` batte sempre tutto: e l'override esplicito.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CONFIG_FILENAMES = ("yt-learner.json", ".yt-learner.json")



class ConfigError(RuntimeError):
    """Preset inesistente o file di configurazione malformato."""


@dataclass
class BrainPreset:
    """Una destinazione di scrittura per le schede di knowledge."""

    name: str
    root: Path                       # radice del vault
    subdir: str = "knowledge"        # sottocartella delle schede
    layout: str = "flat"             # "flat" -> <slug>.md, "dated" -> <YYYY-MM-DD>-<slug>.md
    note_type: str = "reference"     # valore del campo `type` nel frontmatter
    legacy_metadata_type: bool = True  # scrive anche `metadata.type` (compat vault che leggono ancora il campo annidato)
    index: str = ""                  # indice da aggiornare, relativo alla radice
    lint_cmd: list[str] = field(default_factory=list)  # comando di lint post-scrittura
    description: str = ""

    @property
    def target_dir(self) -> Path:
        return self.root / self.subdir if self.subdir else self.root

    @property
    def index_path(self) -> Path | None:
        return self.root / self.index if self.index else None

    def filename(self, slug: str, when: str | None = None) -> str:
        """Nome file secondo il layout del preset."""
        if self.layout == "dated":
            return f"{when or date.today().isoformat()}-{slug}.md"
        return f"{slug}.md"

    def resolved_lint_cmd(self) -> list[str]:
        """Sostituisce i placeholder `{root}` nel comando di lint."""
        return [part.format(root=str(self.root)) for part in self.lint_cmd]


# ------------------------------------------------------------------ built-in


def _manent_lint_cmd() -> list[str]:
    """Comando di lint Manent, solo se la CLI e effettivamente presente.

    `MANENT_CLI` = percorso di `packages/cli/dist/index.js` di un checkout di Manent;
    altrimenti il comando `manent` nel PATH; altrimenti niente lint.
    """
    cli = os.getenv("MANENT_CLI")
    if cli and Path(cli).exists():
        return ["node", cli, "lint", "{root}"]
    if shutil.which("manent"):
        return ["manent", "lint", "{root}"]
    return []


def builtin_presets() -> dict[str, BrainPreset]:
    """Preset disponibili senza configurazione."""
    # senza MANENT_VAULT il vault e la cartella da cui lanci il tool
    manent_root = Path(os.getenv("MANENT_VAULT") or ".")
    return {
        "default": BrainPreset(
            name="default",
            root=Path("./brain"),
            subdir="knowledge",
            layout="flat",
            note_type="reference",
            description="cartella locale del progetto (./brain/knowledge)",
        ),
        "manent": BrainPreset(
            name="manent",
            root=manent_root,
            # `knowledge/` e escluso da .manentignore: le fonti datate vanno in library/
            subdir="library",
            layout="dated",
            note_type="raw-source",
            legacy_metadata_type=False,
            index="MEMORY.md",
            lint_cmd=_manent_lint_cmd(),
            description=f"vault Manent ({manent_root}/library, type raw-source, lint Manent se presente)",
        ),
    }


# ------------------------------------------------------- config utente (JSON)


def _config_files() -> list[Path]:
    candidates = [Path.cwd() / name for name in CONFIG_FILENAMES]
    candidates += [Path.home() / name for name in CONFIG_FILENAMES]
    return [path for path in candidates if path.is_file()]


def _preset_from_dict(name: str, data: dict[str, Any]) -> BrainPreset:
    root = data.get("root") or data.get("path")
    if not root:
        raise ConfigError(f"Il preset {name!r} non definisce 'root'.")
    lint = data.get("lint_cmd") or []
    if isinstance(lint, str):
        lint = lint.split()
    return BrainPreset(
        name=name,
        root=Path(os.path.expandvars(str(root))).expanduser(),
        subdir=str(data.get("subdir", "knowledge")),
        layout=str(data.get("layout", "flat")),
        note_type=str(data.get("note_type", "reference")),
        legacy_metadata_type=bool(data.get("legacy_metadata_type", True)),
        index=str(data.get("index", "")),
        lint_cmd=[str(part) for part in lint],
        description=str(data.get("description", "")),
    )


def load_presets() -> dict[str, BrainPreset]:
    """Built-in + preset definiti dall'utente (questi ultimi vincono)."""
    presets = builtin_presets()
    for path in _config_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Config %s ignorata (%s).", path, exc)
            continue
        entries = data.get("brains", data)
        if not isinstance(entries, dict):
            log.warning("Config %s ignorata: manca l'oggetto 'brains'.", path)
            continue
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            try:
                presets[str(name)] = _preset_from_dict(str(name), entry)
            except ConfigError as exc:
                log.warning("%s", exc)
    return presets


def resolve_brain(
    selector: str | None = None, brain_dir: str | None = None
) -> BrainPreset:
    """Sceglie il preset da usare.

    - `brain_dir` esplicito -> preset ad hoc su quella cartella (eventualmente
      ereditando le convenzioni del preset selezionato);
    - `selector` = nome di preset noto, oppure un percorso;
    - niente -> preset `default`.
    """
    presets = load_presets()
    preset = presets["default"]

    if selector:
        if selector in presets:
            preset = presets[selector]
        else:
            candidate = Path(selector).expanduser()
            if candidate.is_dir() or os.sep in selector or "/" in selector:
                preset = BrainPreset(
                    name="custom",
                    root=candidate,
                    subdir="",
                    description=f"cartella indicata a mano ({candidate})",
                )
            else:
                raise ConfigError(
                    f"Preset di brain sconosciuto: {selector!r}. "
                    f"Disponibili: {', '.join(sorted(presets))}. "
                    "Puoi anche passare direttamente un percorso."
                )

    if brain_dir:
        # override esplicito della cartella, conservando le altre convenzioni
        target = Path(brain_dir).expanduser()
        preset = BrainPreset(
            name=preset.name if selector else "custom",
            root=target,
            subdir="",
            layout=preset.layout if selector else "flat",
            note_type=preset.note_type if selector else "reference",
            legacy_metadata_type=preset.legacy_metadata_type,
            index="",
            lint_cmd=preset.lint_cmd if selector else [],
            description=f"cartella indicata a mano ({target})",
        )
    return preset


def describe_presets() -> str:
    """Testo per `--list-brains`."""
    lines = ["Destinazioni disponibili per il brain:"]
    for name, preset in sorted(load_presets().items()):
        target = preset.target_dir
        layout = "<data>-<slug>.md" if preset.layout == "dated" else "<slug>.md"
        exists = "ok" if preset.root.exists() else "cartella assente"
        lines.append(
            f"  {name:<10} {target}  [{layout}, type={preset.note_type}, {exists}]"
        )
        if preset.description:
            lines.append(f"{'':<13}{preset.description}")
        if preset.index:
            lines.append(f"{'':<13}indice: {preset.index_path}")
        if preset.lint_cmd:
            lines.append(f"{'':<13}lint: {' '.join(preset.resolved_lint_cmd())}")
    lines.append("")
    lines.append(
        "Puoi definirne altri in `yt-learner.json` (cwd) o `~/.yt-learner.json`:\n"
        '  {"brains": {"lavoro": {"root": "D:/vault", "subdir": "fonti", '
        '"layout": "dated", "note_type": "raw-source", "index": "MEMORY.md"}}}'
    )
    return "\n".join(lines)
