from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from feature_extraction import extract_features_from_directory
from visualisation import visualize_clusters
from manual_feature_extraction import extract_features_from_dir_man

def cluster_images(features, n_clusters=3):
    """Clusters images based on their ResNet-50 features using K-Means."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # Normalize

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    return labels, features_scaled

# Example: Clustering images from a folderin

def integrate():
    image_dir = "/home/ansh/outlier-detection/data/random_images"
    features, image_paths = extract_features_from_directory(image_dir)
    n_clusters = 3  # Set number of clusters
    labels, features_scaled = cluster_images(features, n_clusters)

    # Print cluster assignments
    for img, label in zip(image_paths, labels):
        print(f"Image: {img} -> Cluster {label}")

    # Visualize clusters
    visualize_clusters(features_scaled, labels)

integrate()