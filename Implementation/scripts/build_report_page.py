"""Combines report_template.html with the qualitative-example images into the published report
page (the two hero examples from semantic_drift_report.md's Results section: the low-drift surfer
chain and the high-drift baseball-glove chain across all three conditions).

Run from Implementation/: python scripts/build_report_page.py [output_path]
Defaults to writing report_page.html next to this script.
"""

import base64
import io
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = Path(__file__).resolve().parent / "report_template.html"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "report_page.html"

FIGS = {
    "surfer_original": "results/baseline/000000270122_object_level/step0_original.png",
    "surfer_edited": "results/baseline/000000270122_object_level/step4.png",
    "glove_original": "results/baseline/000000407298_object_level/step0_original.png",
    "glove_baseline": "results/baseline/000000407298_object_level/step4.png",
    "glove_region_locking": "results/mitigated/region_locking/000000407298_object_level/step4.png",
    "glove_masked_conditioning": "results/mitigated/masked_conditioning/000000407298_object_level/step4.png",
}
FIG_SIZE = 420
JPEG_QUALITY = 82


def encode(path: Path) -> str:
    img = Image.open(path).convert("RGB").resize((FIG_SIZE, FIG_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT

    figs_b64 = {key: encode(ROOT / rel) for key, rel in FIGS.items()}
    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    def repl(m: re.Match) -> str:
        return f"data:image/jpeg;base64,{figs_b64[m.group(1)]}"

    html, n = re.subn(r"\{\{IMG:(\w+)\}\}", repl, html)
    if n != len(FIGS):
        raise ValueError(f"expected {len(FIGS)} image placeholders, replaced {n}")

    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
