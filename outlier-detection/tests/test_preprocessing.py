import unittest
import os
import shutil
import numpy as np
import cv2
from PIL import Image
import sys
sys.path.append(os.path.abspath("..")) 
from src.preprocessing import preprocess_image

class TestImagePreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "./temp_cv2_test_images"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.image_path = os.path.join(cls.test_dir, "test_img.jpg")
        img = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
        img.save(cls.image_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_preprocess_image_output_shapes(self):
        processed_rgb, enhanced_gray, binary = preprocess_image(self.image_path)

        self.assertIsInstance(processed_rgb, np.ndarray)
        self.assertIsInstance(enhanced_gray, np.ndarray)
        self.assertIsInstance(binary, np.ndarray)

        self.assertEqual(processed_rgb.shape, (256, 256, 3))
        self.assertEqual(enhanced_gray.shape, (256, 256))
        self.assertEqual(binary.shape, (256, 256))

    def test_invalid_image_path(self):
        with self.assertRaises(ValueError):
            preprocess_image("non_existent_image.jpg")

if __name__ == '__main__':
    unittest.main()