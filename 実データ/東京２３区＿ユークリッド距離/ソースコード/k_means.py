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

def weighted_kmeans(X, weights, n_clusters, pnts=None,max_iter=100,initial = False):
    # データポイントの数
    n_samples = X.shape[0]
    # ランダムに初期クラスタ中心を選択
    if initial:
        centroids = pnts
    else:
        random_indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[random_indices]

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

    # コスト関数（目的関数）の計算
    cost_function_value = 0
    for i in range(len(X)):
        cluster_center = centroids[labels[i]]
        cost_function_value += weights[i]*np.sum((X[i] - cluster_center) ** 2)
    
    return centroids, labels, cost_function_value


# コスト関数単体
def cost_function(X,weights,centroids,labels = 0,non_claster = False):
    # labelがない場合
    if non_claster:
        # クラスタの割り当て
        distances = np.array([np.sum(weights[:, np.newaxis] * (X - centroid) ** 2, axis=1) for centroid in centroids])
        labels = np.argmin(distances, axis=0)
    cost_function_value = 0
    for i in range(len(X)):
        cluster_center = centroids[labels[i]]
        cost_function_value += weights[i]*np.sum((X[i] - cluster_center) ** 2)
    return cost_function_value


if __name__ == '__main__':
    # テストデータの生成
    X = np.random.rand(5, 2)
    print("X:",X)
    weights = np.random.rand(5)
    print("weights:",weights)
    weighted_kmeans(X,weights,3)
