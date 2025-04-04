import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from feature_extraction import extract_features

def detect_outlier(file_paths):
    """
    Detects the outlier as the farthest point from the single-cluster centroid.
    Args:
    - features (np.array): Feature vectors of all images (shape: NxD)
    
    Returns:
    - outlier_index (int): Index of the detected outlier
    """
    features = []
    for file in file_paths:
        feature = extract_features(file)
        features.append(feature)
    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(features_scaled)
    centroid = kmeans.cluster_centers_[0]
    distances = np.linalg.norm(features_scaled - centroid, axis=1)
    outlier_index = np.argmax(distances)
    return outlier_index

def detect_outlier_f(features):
    """
    Detects the outlier as the farthest point from the single-cluster centroid.
    Args:
    - features (np.array): Feature vectors of all images (shape: NxD)
    
    Returns:
    - outlier_index (int): Index of the detected outlier
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(features_scaled)
    centroid = kmeans.cluster_centers_[0]
    distances = np.linalg.norm(features_scaled - centroid, axis=1)
    outlier_index = np.argmax(distances)
    return outlier_index