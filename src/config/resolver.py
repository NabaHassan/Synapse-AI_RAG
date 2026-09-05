"""Profile resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ProfileResolution:
    template_id: str
    reason: str

def resolve_profile_template_id(
    kb_id: str,
    collection_name: str,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    config_dir: Optional[Path] = None,
) -> ProfileResolution:
    """Resolve profile template id for a KB using explicit profile-file convention."""
    if config_dir is not None:
        exact_profile_path = config_dir / "profiles" / f"{kb_id}.yaml"
        if exact_profile_path.exists():
            return ProfileResolution(template_id=kb_id, reason=f"exact_kb_id:{kb_id}")
        return ProfileResolution(template_id="default", reason="default")

    return ProfileResolution(template_id="default", reason="default")
