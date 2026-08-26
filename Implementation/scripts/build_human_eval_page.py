"""Combines human_eval_template.html with human_eval/rating_data.json (produced by
prepare_human_eval.py) into a single self-contained HTML page, ready to publish as an Artifact.

Run from Implementation/: python scripts/build_human_eval_page.py [output_path]
Defaults to writing human_eval_page.html next to this script.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = Path(__file__).resolve().parent / "human_eval_template.html"
DATA_PATH = ROOT / "human_eval" / "rating_data.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "human_eval_page.html"


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    data = DATA_PATH.read_text(encoding="utf-8")
    if "</script>" in data:
        raise ValueError("rating_data.json unexpectedly contains '</script>' — cannot safely embed")

    combined = template.replace("/*__CHAIN_DATA_JSON__*/", data)
    output_path.write_text(combined, encoding="utf-8")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
