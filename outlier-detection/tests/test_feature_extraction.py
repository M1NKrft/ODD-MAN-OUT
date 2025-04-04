import unittest
import os
import shutil
import numpy as np
from PIL import Image
import torch
import sys
sys.path.append(os.path.abspath("..")) 
from src.feature_extraction import extract_features, extract_features_from_directory

class TestFeatureExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a temporary directory and some dummy images."""
        cls.test_dir = "./temp_test_images"
        os.makedirs(cls.test_dir, exist_ok=True)

        for i in range(3):
            img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
            img.save(os.path.join(cls.test_dir, f"img_{i}.jpg"))

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary test directory."""
        shutil.rmtree(cls.test_dir)

    def test_extract_features(self):
        """Test if feature vector has correct shape (2048,)"""
        sample_img = os.path.join(self.test_dir, "img_0.jpg")
        features = extract_features(sample_img)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (2048,))

    def test_extract_features_from_directory(self):
        """Test extraction of features from multiple images."""
        features, paths = extract_features_from_directory(self.test_dir)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 3)  # 3 images
        self.assertEqual(features.shape[1], 2048)
        self.assertEqual(len(paths), 3)
        for path in paths:
            self.assertTrue(os.path.isfile(path))

if __name__ == "__main__":
    unittest.main()
