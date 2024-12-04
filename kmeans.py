import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from loader import load_user_data, load_rating_data, load_movie_data
from config import CONFIG

# 1. User x Item 행렬 생성 함수
def create_user_item_matrix(file_path):
    #data = pd.read_csv(file_path, sep="::", header=None, engine="python", names=["UserID", "MovieID", "Rating", "Timestamp"])
    data = load_user_data(CONFIG['rating_path'])
    user_item_matrix = np.zeros((6040, 3952))  # UserID: 1~6040, MovieID: 1~3952
    for row in data.itertuples():
        user_item_matrix[row[1] - 1, row[2] - 1] = row[3]
    return user_item_matrix

# 2. KMeans 클러스터링 함수
def perform_kmeans_clustering(user_item_matrix, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(user_item_matrix)
    return clusters

# 3. TSNE 시각화 함수
def visualize_clusters(user_item_matrix, clusters):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(user_item_matrix)
    fig, ax = plt.subplots()
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.title("t-SNE Clustering Visualization")
    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    return fig