"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from __future__ import annotations

from pathlib import Path

from corelib.paths import ROOT

WORKSPACE_DIR = ROOT


def resolve_workspace_path(path: str | Path) -> Path:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT / candidate).resolve()
