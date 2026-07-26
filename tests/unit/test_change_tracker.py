from __future__ import annotations

import json
from pathlib import Path

from local_rag_assistant.utils.change_tracker import ChangeTracker


def test_initial_state_returns_empty_dict(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path / "change.log")
    assert tracker.get_state() == {}


def test_detect_changes_handles_new_modified_and_deleted_files(tmp_path: Path) -> None:
    state_file = tmp_path / "change.log"
    tracker = ChangeTracker(state_file)

    one = tmp_path / "one.md"
    two = tmp_path / "two.md"
    three = tmp_path / "three.md"
    one.write_text("one", encoding="utf-8")
    two.write_text("two", encoding="utf-8")
    three.write_text("three", encoding="utf-8")

    tracker.update_state([one, two], hashes={str(one): tracker.calculate_hash(one), str(two): tracker.calculate_hash(two)})
    two.write_text("two updated", encoding="utf-8")

    new_files, modified_files, deleted_files = tracker.detect_changes([two, three])

    assert str(three) in new_files
    assert str(two) in modified_files
    assert str(one) in deleted_files


def test_update_state_persists_json(tmp_path: Path) -> None:
    state_file = tmp_path / "change.log"
    tracker = ChangeTracker(state_file)
    source = tmp_path / "doc.md"
    source.write_text("hello", encoding="utf-8")
    file_hash = tracker.calculate_hash(source)

    tracker.update_state([source], hashes={str(source): file_hash})

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert persisted[str(source)]["hash"] == file_hash
    assert "indexed_at" in persisted[str(source)]


def test_purge_state_removes_entries(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path / "change.log")
    source = tmp_path / "doc.md"
    source.write_text("hello", encoding="utf-8")
    tracker.update_state([source], hashes={str(source): tracker.calculate_hash(source)})

    updated = tracker.purge_state([source])

    assert str(source) not in updated
