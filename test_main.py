import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from main import main


class PrelabelTests(unittest.TestCase):
    def test_prelabel_uses_vessel_checkpoint_and_saves_review_proposals(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.pt"
            weights.touch()
            source = root / "images"
            source.mkdir()
            (source / "boat.jpg").touch()
            with patch("sys.argv", ["main.py", "prelabel", "--weights", str(weights), "--source", str(source)]):
                with patch("ultralytics.YOLO") as yolo:
                    model = yolo.return_value
                    model.names = {0: "vessel"}
                    model.predict.return_value = iter([object()])
                    main()
            yolo.assert_called_once_with(str(weights), task="detect")
            options = model.predict.call_args.kwargs
            self.assertEqual(options["source"], str(source.resolve()))
            self.assertEqual(options["conf"], 0.15)
            self.assertTrue(options["save"])
            self.assertTrue(options["save_txt"])
            self.assertFalse(options["save_conf"])
            self.assertFalse(options["exist_ok"])
            self.assertTrue(options["stream"])

    def test_prelabel_rejects_non_vessel_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.pt"
            weights.touch()
            source = root / "images"
            source.mkdir()
            (source / "boat.jpg").touch()
            with patch("sys.argv", ["main.py", "prelabel", "--weights", str(weights), "--source", str(source)]):
                with patch("ultralytics.YOLO") as yolo:
                    yolo.return_value.names = {0: "person", 8: "boat"}
                    with self.assertRaises(SystemExit) as error:
                        main()
                    self.assertEqual(error.exception.code, 2)
                    yolo.return_value.predict.assert_not_called()

    def test_prelabel_rejects_duplicate_image_stems(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.pt"
            weights.touch()
            source = root / "images"
            source.mkdir()
            (source / "boat.jpg").touch()
            (source / "boat.png").touch()
            with patch("sys.argv", ["main.py", "prelabel", "--weights", str(weights), "--source", str(source)]):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
