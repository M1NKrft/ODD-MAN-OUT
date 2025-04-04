import unittest
import numpy as np
import os
from unittest.mock import patch
from src.single_outlier import detect_outlier, detect_outlier_f

class TestOutlierDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up mock file paths and dummy features."""
        cls.file_paths = ["image_1.jpg", "image_2.jpg", "image_3.jpg"]
        cls.dummy_features = np.array([
            [1.0, 2.0, 3.0, 4.0],  # Normal point
            [1.1, 2.1, 3.1, 4.1],  # Normal point
            [10.0, 20.0, 30.0, 40.0]  # Outlier
        ])

    @patch("src.single_outlier.extract_features")
    def test_detect_outlier(self, mock_extract_features):
        """Test detect_outlier with mocked feature extraction."""
        # Mock the extract_features function to return dummy features
        mock_extract_features.side_effect = lambda file: self.dummy_features[self.file_paths.index(file)]

        # Call the function
        outlier_index = detect_outlier(self.file_paths)

        # Assert the outlier is correctly identified
        self.assertEqual(outlier_index, 2)  # The outlier is at index 2

    def test_detect_outlier_f(self):
        """Test detect_outlier_f with precomputed features."""
        # Call the function with dummy features
        outlier_index = detect_outlier_f(self.dummy_features)

        # Assert the outlier is correctly identified
        self.assertEqual(outlier_index, 2)  # The outlier is at index 2

if __name__ == "__main__":
    unittest.main()