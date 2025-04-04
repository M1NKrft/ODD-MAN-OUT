import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.visualisation import visualize_clusters

class TestVisualisation(unittest.TestCase):
    @patch("src.visualisation.PCA")
    @patch("src.visualisation.plt")
    def test_visualize_clusters(self, mock_plt, mock_pca):
        """Test visualize_clusters function."""
        # Mock PCA transformation
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        mock_pca.return_value = mock_pca_instance

        # Mock plt functions
        mock_plt.figure.return_value = MagicMock()
        mock_plt.scatter.return_value = MagicMock()
        mock_plt.colorbar.return_value = MagicMock()

        # Input data
        features_scaled = np.random.rand(3, 2048)  # Mock scaled features
        labels = [0, 1, 0]  # Mock cluster labels

        # Call the function
        output_path = visualize_clusters(features_scaled, labels)

        # Assertions
        mock_pca.assert_called_once_with(n_components=2)
        mock_pca_instance.fit_transform.assert_called_once_with(features_scaled)
        mock_plt.figure.assert_called_once_with(figsize=(8, 6))
        mock_plt.scatter.assert_called_once()
        mock_plt.colorbar.assert_called_once()
        mock_plt.savefig.assert_called_once_with("plot.jpg")
        mock_plt.close.assert_called_once()

        # Check the return value
        self.assertEqual(output_path, "static/plot.jpg")

if __name__ == "__main__":
    unittest.main()
