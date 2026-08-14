"""Download a diverse ~60-image subset of COCO val2017 into data/raw_images/, with a manifest
recording which COCO categories are present in each image (used later to write per-image
edit-chain instructions without having to re-open every image by hand).

Run from Implementation/: python scripts/download_coco_subset.py

Only stdlib is used (urllib, zipfile, json) so no new dependency is needed beyond requirements.txt.
"""

import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/"
ANNOTATION_MEMBER = "annotations/instances_val2017.json"

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / ".cache"
RAW_IMAGES_DIR = ROOT / "data" / "raw_images"
MANIFEST_PATH = ROOT / "data" / "coco_subset_manifest.json"

N_IMAGES = 60
SEED = 42
MIN_ANNOTATIONS = 2
MAX_ANNOTATIONS = 8


def ensure_annotations() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / "instances_val2017.json"
    if target.exists():
        return target

    zip_path = CACHE_DIR / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print(f"Downloading annotations ({ANNOTATIONS_URL}, ~241MB, one-time)...")
        urllib.request.urlretrieve(ANNOTATIONS_URL, zip_path)

    print("Extracting instances_val2017.json...")
    with zipfile.ZipFile(zip_path) as zf, zf.open(ANNOTATION_MEMBER) as src, open(target, "wb") as dst:
        dst.write(src.read())

    zip_path.unlink()
    return target


def select_diverse_images(annotations_path: Path) -> list[dict]:
    with open(annotations_path) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images_by_id = {img["id"]: img for img in coco["images"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # A handful of annotated objects per image gives room for both a target region and
    # non-target regions to check for drift, without an overly cluttered scene.
    candidates = [
        {
            "image_id": image_id,
            "file_name": images_by_id[image_id]["file_name"],
            "categories": sorted({categories[a["category_id"]] for a in anns}),
        }
        for image_id, anns in anns_by_image.items()
        if MIN_ANNOTATIONS <= len(anns) <= MAX_ANNOTATIONS
    ]

    random.Random(SEED).shuffle(candidates)
    return candidates[:N_IMAGES]


def download_images(selected: list[dict]) -> None:
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(selected, 1):
        dest = RAW_IMAGES_DIR / item["file_name"]
        if dest.exists():
            continue
        url = IMAGE_BASE_URL + item["file_name"]
        print(f"[{i}/{len(selected)}] Downloading {item['file_name']}...")
        urllib.request.urlretrieve(url, dest)


def main():
    annotations_path = ensure_annotations()
    selected = select_diverse_images(annotations_path)
    download_images(selected)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nDone: {len(selected)} images in {RAW_IMAGES_DIR}")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
