"""Unit tests for data_loader.py, using temporary files — no real dataset or model calls."""

import json

import pytest

from src.data_loader import load_edit_chains


def _write_instructions(path, entries):
    path.write_text(json.dumps(entries))


def test_loads_chains_when_all_images_present(tmp_path):
    images_dir = tmp_path / "raw_images"
    images_dir.mkdir()
    (images_dir / "a.jpg").write_bytes(b"fake")

    instructions_path = tmp_path / "edit_instructions.json"
    _write_instructions(instructions_path, [{"image_id": "a.jpg", "chain_type": "global", "instructions": ["x"]}])

    chains = load_edit_chains(str(instructions_path), str(images_dir))
    assert len(chains) == 1
    assert chains[0]["image_id"] == "a.jpg"


def test_raises_when_referenced_image_missing(tmp_path):
    images_dir = tmp_path / "raw_images"
    images_dir.mkdir()

    instructions_path = tmp_path / "edit_instructions.json"
    _write_instructions(instructions_path, [{"image_id": "missing.jpg", "chain_type": "global", "instructions": ["x"]}])

    with pytest.raises(FileNotFoundError):
        load_edit_chains(str(instructions_path), str(images_dir))
