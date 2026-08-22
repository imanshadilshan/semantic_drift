"""Unit tests for mitigation.py's pixel-compositing logic, with SAM/CLIP/the diffusion model
mocked out — these test the box math and paste behavior, not the real models.
"""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import src.mitigation as mitigation


@pytest.fixture
def solid_images():
    pre = Image.new("RGB", (100, 100), color=(255, 0, 0))  # red
    post_full = Image.new("RGB", (100, 100), color=(0, 255, 0))  # green
    return pre, post_full


def test_region_locking_keeps_target_box_and_reverts_everything_else(solid_images):
    pre, post_full = solid_images
    target_box = (10, 10, 20, 20)  # x, y, w, h

    with (
        patch.object(mitigation, "get_region_boxes", lambda img: {"0": target_box}),
        patch.object(mitigation, "crop_regions", lambda img, boxes: dict.fromkeys(boxes)),
        patch.object(mitigation, "identify_target_regions", lambda regions, instr: {"0"}),
        patch.object(mitigation, "run_edit", lambda img, instr: post_full.resize(img.size)),
    ):
        result = np.array(mitigation.region_locking(pre, "some instruction"))

    assert tuple(result[15, 15]) == (0, 255, 0)  # inside target box -> edited
    assert tuple(result[15, 29]) == (0, 255, 0)  # last column still inside the box
    assert tuple(result[15, 31]) == (255, 0, 0)  # just outside the box -> reverted
    assert tuple(result[50, 50]) == (255, 0, 0)  # far outside -> reverted


def test_masked_conditioning_only_touches_padded_crop(solid_images):
    pre, _ = solid_images
    target_box = (0, 0, 10, 10)  # near the corner, to exercise the padding clamp
    padding = 32

    with (
        patch.object(mitigation, "get_region_boxes", lambda img: {"0": target_box}),
        patch.object(mitigation, "crop_regions", lambda img, boxes: dict.fromkeys(boxes)),
        patch.object(mitigation, "identify_target_regions", lambda regions, instr: {"0"}),
        patch.object(mitigation, "run_edit", lambda img, instr: Image.new("RGB", img.size, color=(0, 255, 0))),
    ):
        result = np.array(mitigation.masked_conditioning(pre, "some instruction", padding=padding))

    # expected crop: x0=max(0-32,0)=0, y0=0, x1=min(0+10+32,100)=42, y1=42
    assert tuple(result[10, 10]) == (0, 255, 0)  # inside padded crop -> edited
    assert tuple(result[41, 41]) == (0, 255, 0)  # last row/col of the crop
    assert tuple(result[42, 42]) == (255, 0, 0)  # just outside the crop -> untouched
    assert tuple(result[99, 99]) == (255, 0, 0)  # far outside -> untouched
