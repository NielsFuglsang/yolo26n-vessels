import argparse
from pathlib import Path

from prepare import SOURCES, prepare_dataset


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
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
