import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def detect_outlier(features):
    """
    Detects the outlier as the farthest point from the single-cluster centroid.
    Args:
    - features (np.array): Feature vectors of all images (shape: NxD)
    
    Returns:
    - outlier_index (int): Index of the detected outlier
    """
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Fit a single-cluster KMeans model
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(features_scaled)
    
    # Get cluster center
    centroid = kmeans.cluster_centers_[0]
    
    # Compute distances of each point from the centroid
    distances = np.linalg.norm(features_scaled - centroid, axis=1)
    
    # Find the farthest point
    outlier_index = np.argmax(distances)
    return outlier_index