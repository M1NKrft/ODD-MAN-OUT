import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.clustering import cluster_images

class TestClustering(unittest.TestCase):
    @patch("src.clustering.extract_features_from_directory")
    @patch("src.clustering.visualize_clusters")
    @patch("src.clustering.shutil.copy")
    @patch("src.clustering.os.makedirs")
    @patch("src.clustering.shutil.rmtree")
    def test_cluster_images(self, mock_rmtree, mock_makedirs, mock_copy, mock_visualize_clusters, mock_extract_features):
        """Test the cluster_images function."""
        mock_features = np.array([
            [1000, 2000, 3000, 4000],
            [1100, 2100, 3100, 4100],
            [-1000, -2000, -3000, -4000],  
            [-1100, -2100, -3100, -4100],  
            [5000, 6000, 7000, 8000]  
        ])
        mock_image_paths = [f"image_{i}.jpg" for i in range(5)]
        mock_extract_features.return_value = (mock_features, mock_image_paths)

        # Mock the output of visualize_clusters
        mock_visualize_clusters.return_value = "static/plot.jpg"

        # Input parameters
        image_dir = "test_images"
        output_dir = "test_output"
        n_clusters = 3

        # Call the function
        labels, image_paths, plot_path = cluster_images(image_dir, output_dir, n_clusters)

        # Scale the mock features to match the scaled features passed to visualize_clusters
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(mock_features)

        # Assertions
        mock_extract_features.assert_called_once_with(image_dir)
        mock_rmtree.assert_called_once_with(output_dir, ignore_errors=True)
        mock_makedirs.assert_any_call(output_dir, exist_ok=True)
        self.assertEqual(len(labels), 5)  # 5 images should have labels
        self.assertEqual(len(image_paths), 5)  # 5 image paths should be returned
        self.assertEqual(plot_path, "static/plot.jpg")  # Plot path should match the mock return value

        # Verify that visualize_clusters was called with the scaled features
        mock_visualize_clusters.assert_called_once()
        np.testing.assert_array_almost_equal(
            mock_visualize_clusters.call_args[0][0], scaled_features
        )  # Check scaled features
        self.assertTrue(
            np.array_equal(mock_visualize_clusters.call_args[0][1], labels)
        )  # Check labels

if __name__ == "__main__":
    unittest.main()