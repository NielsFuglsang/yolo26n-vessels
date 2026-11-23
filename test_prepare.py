import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from prepare import prepare_dataset, yolo_box


class PrepareTests(unittest.TestCase):
    def test_coco_boxes_are_clipped_before_normalizing(self):
        self.assertEqual(yolo_box([-10, 5, 30, 20], 100, 50), "0 0.1 0.3 0.2 0.4")
        self.assertIsNone(yolo_box([120, 0, 10, 10], 100, 50))

    def test_local_export_keeps_images_and_labels_paired(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for split, color in (("train", "red"), ("val", "blue")):
                images = root / "local" / "images" / split
                labels = root / "local" / "labels" / split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                Image.new("RGB", (20, 20), color).save(images / "boat.png")
                (labels / "boat.txt").write_text("0 0.5 0.5 0.5 0.5", encoding="utf-8")
            output = root / "export"
            yaml = prepare_dataset(output, [], root / "local", None)
            self.assertEqual(yaml.read_text(), "train: images/train\nval: images/val\nnames:\n  0: vessel\n")
            self.assertTrue((output / "images" / "val" / "local_000000.png").is_file())
            self.assertEqual(
                (output / "labels" / "val" / "local_000000.txt").read_text(),
                "0 0.5 0.5 0.5 0.5",
            )

    def test_unlabelled_image_does_not_publish_a_dataset(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "local" / "images" / "train"
            images.mkdir(parents=True)
            Image.new("RGB", (20, 20)).save(images / "unlabelled.png")
            output = root / "export"
            with self.assertRaisesRegex(ValueError, "Missing label"):
                prepare_dataset(output, [], root / "local", None)
            self.assertFalse(output.exists())
