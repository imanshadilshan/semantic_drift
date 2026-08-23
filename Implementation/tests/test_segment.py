"""Unit tests for segment.py's box handling, with SAM's mask generator mocked out.

Regression test for a real bug: SAM's own "bbox" values are Python floats (from a tensor
.tolist()), and PIL.Image.paste() raises TypeError on a non-integer box — crop() tolerates
floats silently, which is why this wasn't caught until mitigation.py's paste() calls ran
against real SAM output on Colab.
"""

from unittest.mock import MagicMock, patch

from PIL import Image

import src.segment as segment


def test_get_region_boxes_casts_float_bboxes_to_int():
    fake_generator = MagicMock()
    fake_generator.generate.return_value = [
        {"bbox": [10.4, 20.6, 30.0, 40.2]},
        {"bbox": [0.0, 0.0, 5.5, 5.5]},
    ]

    image = Image.new("RGB", (64, 64))
    with patch.object(segment, "_get_mask_generator", return_value=fake_generator):
        boxes = segment.get_region_boxes(image)

    assert boxes == {"0": (10, 21, 30, 40), "1": (0, 0, 6, 6)}
    for box in boxes.values():
        assert all(isinstance(v, int) for v in box)


def test_paste_does_not_raise_on_get_region_boxes_output():
    fake_generator = MagicMock()
    fake_generator.generate.return_value = [{"bbox": [10.4, 20.6, 30.0, 40.2]}]

    image = Image.new("RGB", (64, 64))
    with patch.object(segment, "_get_mask_generator", return_value=fake_generator):
        boxes = segment.get_region_boxes(image)

    x, y, w, h = boxes["0"]
    patch_img = Image.new("RGB", (w, h))
    image.paste(patch_img, (x, y))  # would raise TypeError before the fix
