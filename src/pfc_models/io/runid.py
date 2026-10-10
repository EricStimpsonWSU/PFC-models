"""Run identifiers: YYYYMMDD-<model>-<8 chars of config hash>.

The hash is over the canonical serialization of the config, so the same config
re-run on the same day lands in the same directory and a changed parameter does
not.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Mapping


def config_hash(config: Mapping[str, Any], length: int = 8) -> str:
    """Stable hash of a config mapping. Key order does not matter."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(blob.encode()).hexdigest()[:length]


def run_id(model: str, config: Mapping[str, Any], date: _dt.date | None = None) -> str:
    """`20260914-logpfc_extv-9f3a1c42`."""
    day = (date or _dt.date.today()).strftime("%Y%m%d")
    return f"{day}-{model}-{config_hash(config)}"
