import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_clusters(features_scaled, labels):
    """Reduces feature dimensions using PCA and visualizes clusters."""
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="viridis", edgecolors='k')
    plt.colorbar(scatter, label="Cluster Label")
    plt.title("Image Clustering with ResNet-50 Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig("plot.jpg")
    plt.close()
    return "static/plot.jpg"