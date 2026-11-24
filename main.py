import argparse
from pathlib import Path

from prepare import IMAGE_EXTENSIONS, SOURCES, prepare_dataset


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def confidence(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data and fine-tune YOLO26n for vessel detection.")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Export Hugging Face and/or local YOLO data")
    prepare.add_argument("--source", nargs="+", choices=[*SOURCES, "none"], default=["singapore"])
    prepare.add_argument("--local", type=Path, help="Local YOLO export with images/ and labels/")
    prepare.add_argument("--output", type=Path, default=Path("data") / "vessels")
    prepare.add_argument("--limit", type=positive_int, help="Smoke-test row cap per remote source and split")

    train = commands.add_parser("train", help="Fine-tune from pretrained detection weights")
    train.add_argument("--weights", default="yolo26n.pt")
    train.add_argument("--epochs", type=positive_int, default=50)
    train.add_argument("--project", type=Path, default=Path("runs"))
    train.add_argument("--name", default="yolo26n-vessels")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)

    evaluate = commands.add_parser("evaluate", help="Measure checkpoint precision, recall and mAP")
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    prelabel = commands.add_parser("prelabel", help="Generate vessel box proposals for manual review")
    prelabel.add_argument("--weights", type=Path, required=True, help="Fine-tuned vessel checkpoint")
    prelabel.add_argument("--source", type=Path, default=Path("data") / "examples", help="Flat image folder")
    prelabel.add_argument("--conf", type=confidence, default=0.15)
    prelabel.add_argument("--project", type=Path, default=Path("runs"))
    prelabel.add_argument("--name", default="prelabel")
    prelabel.add_argument("--device", default="cpu", help="For example: cpu or 0")
    prelabel.add_argument("--imgsz", type=positive_int, default=640)
    for command in (train, evaluate):
        command.add_argument("--data", type=Path, default=Path("data") / "vessels" / "dataset.yaml")
        command.add_argument("--device", default="cpu", help="For example: cpu or 0")
        command.add_argument("--batch", type=positive_int, default=4)
        command.add_argument("--imgsz", type=positive_int, default=640)

    args = parser.parse_args()
    if args.command == "prepare":
        if len(set(args.source)) != len(args.source) or ("none" in args.source and args.source != ["none"]):
            prepare.error("select unique sources; 'none' must be used alone")
        try:
            output = prepare_dataset(args.output, [] if args.source == ["none"] else args.source, args.local, args.limit)
        except (ValueError, FileExistsError) as error:
            prepare.error(str(error))
        print(f"Dataset ready: {output}")
        return

    if args.command == "prelabel":
        if not args.weights.is_file():
            prelabel.error(f"Checkpoint not found: {args.weights}")
        if not args.source.is_dir():
            prelabel.error(f"Image directory not found: {args.source}")
        entries = list(args.source.iterdir())
        if any(entry.is_dir() for entry in entries):
            prelabel.error("Image directory must be flat; move nested images into a separate folder.")
        if any(entry.is_file() and entry.suffix.lower() not in IMAGE_EXTENSIONS for entry in entries):
            prelabel.error("Image directory must contain only JPG or PNG files.")
        images = [entry for entry in entries if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            prelabel.error(f"No JPG or PNG images found: {args.source}")
        if len({image.stem.casefold() for image in images}) != len(images):
            prelabel.error("Images must have unique filenames without extensions.")

        from ultralytics import YOLO

        model = YOLO(str(args.weights), task="detect")
        if model.names != {0: "vessel"}:
            prelabel.error("Checkpoint must be a single-class detector with class 0 named 'vessel'.")
        for _ in model.predict(
            source=str(args.source.resolve()), device=args.device, imgsz=args.imgsz, conf=args.conf,
            project=str(args.project.resolve()), name=args.name, exist_ok=False,
            save=True, save_txt=True, save_conf=False, stream=True,
        ):
            pass
        return

    from ultralytics import YOLO

    if not args.data.is_file():
        parser.error(f"Dataset YAML not found: {args.data}. Run prepare first.")
    options = dict(data=str(args.data.resolve()), device=args.device, batch=args.batch, imgsz=args.imgsz)
    if args.command == "train":
        if args.workers < 0:
            train.error("--workers cannot be negative")
        model = YOLO(args.weights, task="detect")
        model.train(
            **options, epochs=args.epochs, project=str(args.project.resolve()), name=args.name,
            workers=args.workers, seed=args.seed, deterministic=True,
        )
    else:
        if not args.weights.is_file():
            evaluate.error(f"Checkpoint not found: {args.weights}")
        model = YOLO(str(args.weights), task="detect")
        model.val(**options, split=args.split, workers=0, project=str(Path("runs").resolve()))


if __name__ == "__main__":
    main()
