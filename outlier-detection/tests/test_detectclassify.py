import unittest
import os
import numpy as np
import sys
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from detectclassify import detect_and_classify

class TestDetectAndClassify(unittest.TestCase):
    def setUp(self):
        self.image_dir = "/home/ansh/ODD-MAN-OUT/outlier-detection/images"
        self.model_path = "/home/ansh/flowerz/flower_model.pth"
        self.expected_odd_index = 3 

    @patch("detectclassify.classify_image")
    def test_detect_and_classify(self, mock_classify):
        mock_classify.return_value = "Giant White Arum Lily"

        odd_path, predicted_name, other_img_path = detect_and_classify(
            odd_data_folder=self.image_dir,
            model_path=self.model_path
        )
        self.assertTrue(os.path.exists(odd_path))
        self.assertEqual(predicted_name, "Giant White Arum Lily")
        self.assertTrue(os.path.exists(other_img_path))

        files = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.endswith(("jpg", "png"))
        ])
        expected_odd_path = files[self.expected_odd_index]
        self.assertEqual(odd_path, expected_odd_path)


if __name__ == "__main__":
    unittest.main()