import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, euclidean
import math

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
        distances = np.array([np.sum((X - centroid) ** 2, axis=1) for centroid in centroids])
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

def weighted_kmedians(X, weights, n_clusters, pnts=None, max_iter=100, initial = False):
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
        new_centroids = []
        for k in range(n_clusters):
            new_centroids.append(geometric_median(X[labels == k], weights=weights[labels == k]))
        new_centroids =  np.array(new_centroids)

        # 収束チェック
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # コスト関数（目的関数）の計算
    cost_function_value = 0
    for i in range(len(X)):
        cluster_center = centroids[labels[i]]
        cost_function_value += weights[i]*math.sqrt(np.sum((X[i] - cluster_center)**2))
    
    return centroids, labels, cost_function_value

def geometric_median(X, mesh_weight, eps=1e-5):
    #初期点は平均値から始める
    if X.size == 0:
        print("X.size is 0")
        return ["None"]
    y = np.mean(X, 0)
    mesh_weight = mesh_weight.reshape([-1,1])
    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]
        zero = (D == 0)[:, 0]
        Dinv = mesh_weight[nonzeros] / D[nonzeros]
        Dinvs = np.sum(Dinv)
        #重みが0のものにだけ当たっちゃった場合
        if Dinvs == 0:
            print("Dinvs == 0")
            return y
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        # yとx1,...,xmが一つも被っていない場合
        if num_zeros == 0:
            y1 = T
        # yとx1,...,xmが全て被っている→つまり全部同じ点
        elif num_zeros == len(X):
            return y
        # 1点だけ被っている（全てのサンプル点が異なる座標を持つという仮定を入れている）
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else mesh_weight[zero]/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y
        # 閾値を下回った時に終了
        if euclidean(y, y1) < eps:
            return y1

        y = y1
        

# コスト関数単体
def cost_function(X,weights,centroids,labels = 0,non_claster = False,median = False):
    # labelがない場合
    if non_claster:
        # クラスタの割り当て
        distances = np.array([np.sum(weights[:, np.newaxis] * (X - centroid) ** 2, axis=1) for centroid in centroids])
        labels = np.argmin(distances, axis=0)
    cost_function_value = 0
    if median:
        for i in range(len(X)):
            cluster_center = centroids[labels[i]]
            cost_function_value += weights[i]*math.sqrt(np.sum((X[i] - cluster_center) ** 2))
    else:
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
