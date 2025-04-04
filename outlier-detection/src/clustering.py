import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from feature_extraction import extract_features_from_directory
from visualisation import visualize_clusters

def cluster_images(image_dir, output_dir, n_clusters=3):
    """Clusters images and saves them into separate folders."""

    features, image_paths = extract_features_from_directory(image_dir)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    
    class_folders = {}
    for i in range(n_clusters):
        class_folder = os.path.join(output_dir, f"class-{i}")
        os.makedirs(class_folder, exist_ok=True)
        class_folders[i] = class_folder
    for img_path, label in zip(image_paths, labels):
        shutil.copy(img_path, os.path.join(class_folders[label], os.path.basename(img_path)))

    s = visualize_clusters(features_scaled, labels)

    return labels, image_paths, s