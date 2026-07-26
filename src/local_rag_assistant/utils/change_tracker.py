"""Incremental change tracking with JSON persistence."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ChangeTracker:
    """Track new, modified, and deleted source files across indexing runs."""

    STATE_FILE = "data/cache/change.log"

    def __init__(self, state_file: str | Path | None = None) -> None:
        self.state_file = Path(state_file or self.STATE_FILE)

    def get_state(self) -> dict[str, dict[str, Any]]:
        if not self.state_file.exists():
            return {}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid change tracker state file: {self.state_file}") from exc

    def detect_changes(self, files: list[str | Path]) -> tuple[list[str], list[str], list[str]]:
        state = self.get_state()
        normalized_files = [str(Path(file)) for file in files]

        new_files: list[str] = []
        modified_files: list[str] = []
        deleted_files = sorted(set(state).difference(normalized_files))

        for file_path in normalized_files:
            current_hash = self.calculate_hash(file_path)
            current_mtime = Path(file_path).stat().st_mtime
            previous = state.get(file_path)
            if previous is None:
                new_files.append(file_path)
                continue
            if previous.get("hash") != current_hash or previous.get("mtime") != current_mtime:
                modified_files.append(file_path)

        return sorted(new_files), sorted(modified_files), deleted_files

    def update_state(self, files: list[str | Path], hashes: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
        state = self.get_state()
        timestamp = datetime.now(timezone.utc).isoformat()

        for raw_path in files:
            path = Path(raw_path)
            normalized = str(path)
            state[normalized] = {
                "mtime": path.stat().st_mtime,
                "hash": (hashes or {}).get(normalized) or self.calculate_hash(path),
                "indexed_at": timestamp,
            }

        self._write_state(state)
        return state

    def purge_state(self, deleted_sources: list[str | Path]) -> dict[str, dict[str, Any]]:
        state = self.get_state()
        for source in deleted_sources:
            state.pop(str(Path(source)), None)
        self._write_state(state)
        return state

    def calculate_hash(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot hash missing file: {path}")

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_state(self, state: dict[str, dict[str, Any]]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["ChangeTracker"]
