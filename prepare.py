import hashlib
import json
import math
import shutil
from collections import Counter
from collections.abc import Sequence
from io import BytesIO
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

SOURCES = {
    "singapore": (
        "ARG-NCTU/Singapore_Maritime_Dataset_coco",
        "5520b959b239f0078ff69bbae8710ffa25f33789",
    ),
    "split-port": (
        "ARG-NCTU/Split_Port_Ship_Classification_Dataset_coco",
        "77509e3f6c55150e964625b6ed7514f73904645d",
    ),
}
SPLITS = {"train": "train", "val": "validation", "test": "test"}
BOATS = {
    "boat", "ferry", "kayak", "sail boat", "speed boat", "vessel/ship",
    "fishing trawler", "high speed craft", "large ferry", "large passenger ship",
    "medium ferry", "motorboat", "pleasure yacht", "sailing boat", "small craft",
    "small fishing boat", "small passenger ship", "speed craft",
}
NON_BOATS = {"buoy", "float/ball", "flying bird/plane", "swimming person", "other"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def yolo_box(bbox: Sequence[float], width: int, height: int) -> str | None:
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"Invalid COCO bounding box: {bbox!r}")
    x, y, w, h = bbox
    left, top = max(x, 0), max(y, 0)
    right, bottom = min(x + w, width), min(y + h, height)
    if right <= left or bottom <= top:
        return None
    cx, cy = (left + right) / (2 * width), (top + bottom) / (2 * height)
    return f"0 {cx:.10g} {cy:.10g} {(right-left)/width:.10g} {(bottom-top)/height:.10g}"


def remote_split(
    source: str, split: str, target: Path, limit: int | None
) -> Counter[str]:
    from datasets import Image as HFImage
    from datasets import load_dataset

    repo, revision = SOURCES[source]
    rows = load_dataset(repo, "default", split=SPLITS[split], revision=revision, streaming=True)
    rows = rows.cast_column("image", HFImage(decode=False))
    stats: Counter[str] = Counter()
    for index, row in enumerate(islice(rows, limit)):
        stem = f"{source}_{index:06d}"
        width, height = row["width"], row["height"]
        if width <= 0 or height <= 0:
            raise ValueError(f"{repo} image {row['image_id']}: invalid image dimensions")
        boxes = []
        for category, bbox in zip(row["objects"]["category"], row["objects"]["bbox"], strict=True):
            name = category.strip().casefold()
            stats[f"source category: {name}"] += 1
            if name in NON_BOATS:
                continue
            if name not in BOATS:
                raise ValueError(f"{repo} {SPLITS[split]} image {row['image_id']}: unmapped category {category!r}")
            box = yolo_box(bbox, width, height)
            if box is None:
                stats["empty boxes dropped"] += 1
            else:
                boxes.append(box)

        payload = row["image"]["bytes"]
        with Image.open(BytesIO(payload)) as image:
            if image.size != (width, height):
                raise ValueError(f"{repo} image {row['image_id']}: image and label dimensions differ")
            suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(image.format or "")
            if suffix is None or image.getexif().get(274, 1) != 1:
                suffix = ".png"
                image.convert("RGB").save(target / "images" / split / (stem + suffix))
            else:
                (target / "images" / split / (stem + suffix)).write_bytes(payload)
        (target / "labels" / split / (stem + ".txt")).write_text("\n".join(boxes), encoding="utf-8")
        stats.update({"images": 1, "vessel boxes": len(boxes)})
    return stats


def local_split(
    root: Path, split: str, target: Path, seen: dict[bytes, Path]
) -> Counter[str]:
    stats: Counter[str] = Counter()
    images = root / "images" / split
    if not images.is_dir():
        return stats
    stems = set()
    for index, image in enumerate(sorted(images.iterdir())):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if image.stem.casefold() in stems:
            raise ValueError(f"{image}: another image in this split has the same stem")
        stems.add(image.stem.casefold())
        label = root / "labels" / split / (image.stem + ".txt")
        if not label.is_file():
            raise ValueError(f"Missing label: {label}. Use an empty file for a confirmed negative.")
        lines = label.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError(f"{label}:{number}: expected class 0 and four normalized coordinates")
            x, y, w, h = map(float, fields[1:])
            if (
                not all(math.isfinite(value) and 0 <= value <= 1 for value in (x, y, w, h))
                or w <= 0 or h <= 0
                or x - w / 2 < -1e-6 or y - h / 2 < -1e-6
                or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6
            ):
                raise ValueError(f"{label}:{number}: invalid normalized box")
        with image.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").digest()
        if digest in seen:
            raise ValueError(f"Duplicate local image: {image} matches {seen[digest]}")
        seen[digest] = image
        stem = f"local_{index:06d}"
        shutil.copyfile(image, target / "images" / split / (stem + image.suffix.lower()))
        shutil.copyfile(label, target / "labels" / split / (stem + ".txt"))
        stats.update({"images": 1, "vessel boxes": len(lines)})
    return stats


def prepare_dataset(
    output: Path, sources: Sequence[str], local: Path | None, limit: int | None
) -> Path:
    if not sources and local is None:
        raise ValueError("Select at least one source or supply --local.")
    if local is not None and not (local / "images").is_dir():
        raise ValueError(f"Expected a YOLO export with an images directory: {local}")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a fresh directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[bytes, Path] = {}
    summary = {}
    yaml = ""
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temp:
        target = Path(temp) / "dataset"
        local_images = 0
        for split in SPLITS:
            for kind in ("images", "labels"):
                (target / kind / split).mkdir(parents=True)
            counts = {source: remote_split(source, split, target, limit) for source in sources}
            if local is not None:
                counts["local"] = local_split(local, split, target, seen)
                local_images += counts["local"]["images"]
            total = sum(counts.values(), Counter())
            print(f"{split}: {total['images']} images, {total['vessel boxes']} vessel boxes", flush=True)
            if split != "test" and not total["vessel boxes"]:
                raise ValueError(f"{split} must contain at least one vessel box")
            if total["images"]:
                yaml += f"{split}: images/{split}\n"
            summary[split] = counts
        if local is not None and not local_images:
            raise ValueError(f"No supported images found in the local export: {local}")
        (target / "dataset.yaml").write_text(yaml + "names:\n  0: vessel\n", encoding="utf-8")
        manifest = {"sources": {name: SOURCES[name] for name in sources}, "limit": limit, "counts": summary}
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        target.rename(output)
    return output / "dataset.yaml"
