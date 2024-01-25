import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def kmeans(pnts, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pnts)
    #クラスタの中心点
    centroids = kmeans.cluster_centers_
    # 各データポイントのクラスタラベル
    labels = kmeans.labels_
    # 結果のプロット
    plt.scatter(pnts[:, 0], pnts[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
    plt.show()

def weighted_kmeans(X, weights, pnts = 0, n_clusters, max_iter=100):
    # データポイントの数
    n_samples = X.shape[0]

    # ランダムに初期クラスタ中心を選択 もしくは初期点を入力
    if pnts == 0:
        random_indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[random_indices]
    else :
        centroids = pnts
        
    for _ in range(max_iter):
        # クラスタの割り当て
        distances = np.array([np.sum(weights[:, np.newaxis] * (X - centroid) ** 2, axis=1) for centroid in centroids])
        labels = np.argmin(distances, axis=0)
        # 新しいクラスタの中心を計算
        new_centroids = np.array([np.average(X[labels == k], axis=0, weights=weights[labels == k]) for k in range(n_clusters)])

        # 収束チェック
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


if __name__ == '__main__':
    # テストデータの生成
    X = np.random.rand(5, 2)
    print("X:",X)
    weights = np.random.rand(5)
    print("weights:",weights)
    weighted_kmeans(X,weights,3)
