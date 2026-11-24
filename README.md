# Maritime vessel detection with YOLO26n

Fine-tune a pretrained [Ultralytics YOLO26n](https://docs.ultralytics.com/models/yolo26/) detector for boats in maritime imagery. Class `0` is `vessel`: it includes sailboats, motorboats, ferries, and larger ships. The model predicts bounding boxes, not vessel types.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (0.10.4 or newer). From the repository directory on 64-bit Windows or Linux, install the locked Python 3.13 environment:

```shell
uv sync --locked --extra cpu
```

For an NVIDIA GPU with a driver that supports CUDA 12.8, use `uv sync --locked --extra cuda` **instead** and pass `--device 0` when training or evaluating. Check the GPU is visible with `uv run --no-sync python -c "import torch; print(torch.cuda.is_available())"`. Keep `--no-sync` on run commands so uv does not change the selected environment.

Try a one-epoch CPU check first. `--limit` takes the first rows of each source split, so this is for checking setup, not measuring accuracy:

```shell
uv run --no-sync python main.py prepare --source singapore --limit 12 --output data/vessels-smoke
uv run --no-sync python main.py train --data data/vessels-smoke/dataset.yaml --epochs 1 --imgsz 320 --batch 2 --name smoke
```

## Train and evaluate

1. **Prepare the data.** The default is `singapore`; add `split-port` to combine both datasets. Preparation preserves the published train, validation, and test splits, converts COCO boxes to normalized YOLO rows, and reports category and box counts in `manifest.json`. Pick a fresh output directory to rebuild. The two datasets total about 7.7 GB upstream; allow additional space for the prepared images, dependencies, and training runs.
2. **Fine-tune.** The first run downloads `yolo26n.pt`. On a small GPU start with `--batch 2` (or `--batch 1` if needed). The best checkpoint is saved in `runs/yolo26n-vessels/weights/best.pt` on the first run; Ultralytics increments the directory name on repeats.
3. **Evaluate.** Tune on validation data. Check test precision, recall, and mAP only after deciding on the training settings. The published splits may contain visually similar frames; for a better estimate on new footage, reserve your own test images from entirely different clips or scenes. Inspect predictions on representative images, especially small vessels and difficult lighting.

Training uses a balanced marine augmentation profile by default. It combines moderate HSV and brightness/contrast changes, CLAHE and gamma correction, mild blur and compression, geometric transforms, Mosaic, and mild grid distortion as a fish-eye-lens approximation. Validation and test data are not augmented. Use `--augmentation-profile none` to run a clean baseline:

```shell
uv run --no-sync python main.py train --epochs 50 --augmentation-profile balanced
uv run --no-sync python main.py train --epochs 50 --augmentation-profile none
```

```shell
uv run --no-sync python main.py prepare --source singapore split-port
uv run --no-sync python main.py train --device 0 --epochs 50 --batch 2
uv run --no-sync python main.py evaluate --weights runs/yolo26n-vessels/weights/best.pt --device 0
```

Use `--device cpu` (or omit the option for the Python commands) if no GPU is available. If you did not reserve a test split for a local-only dataset, use `evaluate --split val` instead and treat it as a validation result, not a held-out test.

## Pre-label examples for review

After training on both sources above, use the actual best checkpoint from the training run to propose boxes on a flat folder of images:

```shell
uv run --no-sync python main.py prelabel --weights runs/yolo26n-vessels/weights/best.pt --source data/examples --device 0
```

Omit `--device 0` for CPU. If the training directory was incremented, adjust `--weights` accordingly. The checkpoint must be a single-class detector with class `0` named `vessel`. The input folder must contain only JPG or PNG files with unique stems (for example, do not mix `frame.jpg` and `frame.png`).

Ultralytics saves annotated JPG previews in `runs/prelabel/` and YOLO-format proposals in its `labels/` subfolder. Repeated runs use an incremented directory, such as `runs/prelabel-2/`, rather than overwriting earlier results. The original images are not modified. The default `--conf 0.15` favors more proposals for review; change it to adjust the precision/recall tradeoff.

These are **unreviewed proposals**, not ground truth. Review every original image, including images without a label file: no file means no detection, not a confirmed negative. Correct or remove false boxes and add missed boats. Only create an empty label after confirming there are no vessels. Keep related frames together when splitting, and manually label a held-out set from different scenes before using your examples for training.

## Add your own images

Keep raw examples under `data/examples/` and review proposals with your preferred annotation tool. Export reviewed original images and labels in this layout (do not use the annotated previews as training images):

```text
data/my-yolo/
  images/train/frame.jpg
  labels/train/frame.txt
  images/val/other-frame.jpg
  labels/val/other-frame.txt
  # Optional: images/test/ and labels/test/
```

Each label row is `0 x_center y_center width height`, normalized to the image dimensions. Use an empty `.txt` only for an image confirmed to contain no vessels. Only flat split folders with `.jpg`, `.jpeg`, or `.png` images are imported. Keep frames from the same clip or scene together to avoid inflated validation scores; the importer rejects byte-identical local images but does not detect near duplicates.

```shell
uv run --no-sync python main.py prepare --source singapore split-port --local data/my-yolo --output data/vessels-with-local
uv run --no-sync python main.py train --data data/vessels-with-local/dataset.yaml --device 0
```

Use `--source none --local data/my-yolo` to train entirely on your own labelled data. Train and validation must each contain a labelled vessel.

## Sources and publishing

- [Singapore Maritime Dataset (COCO)](https://huggingface.co/datasets/ARG-NCTU/Singapore_Maritime_Dataset_coco)
- [Split Port Ship Classification Dataset (COCO)](https://huggingface.co/datasets/ARG-NCTU/Split_Port_Ship_Classification_Dataset_coco)

The download revisions are pinned in [prepare.py](./prepare.py). Commit the code and `uv.lock`, not datasets, images, checkpoints, or run outputs. Check the original dataset terms and [Ultralytics licensing](https://www.ultralytics.com/license) before redistributing data or trained weights.
