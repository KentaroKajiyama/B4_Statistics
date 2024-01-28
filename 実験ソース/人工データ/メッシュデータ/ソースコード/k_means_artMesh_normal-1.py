# coding: utf-8
import os
import numpy as np
import math
from scipy.spatial import Voronoi
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from turfpy.measurement import boolean_point_in_polygon
from geojson import Feature, Point
from pathlib import Path
from datetime import datetime
from matplotlib import rcParams
rcParams['lines.markersize'] = 1.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio','Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

"""
問題設定① 
n個の施設配置、最適な配置は総平均（期待値）で評価する。
----------------------------------------------------------------
仮定
・10×10の正方形領域
・人口分布は二次元正規分布
・人々はもっとも近いポストに手紙を出す
・距離は直線距離で近似する
・k-means法を利用
・サンプルは無限である
・正規化定数は１とみなす
----------------------------------------------------------------
プログラムの改善点
・一点一点独立に扱っているので統一性を持たせたい
・挙動を見たいので更新過程も可視化する
・可視化になるべく時間がかからないようにしたい
・座標系を統一してそのままGIS上でも扱えるようにしたい。
・
"""

################################################################
# 環境変数の設定
# ディレクトリ名の環境変数
Appendix = 0 ; sec1 = 1 ; secLocation = 2 ; secPrincipal = 3 ; secMedian = 4
secExperiment = 5 ; secAnalysis = 6 ; secSummary = 7
# 分布選択の環境変数
RandomMesh = 0 ; NormalMesh = 1 
# 正規分布のパラメータ
mu1 = [0,0]; sigma1 = [[1,0],[0,1]]
mu2 = [0,0]; sigma2= [[2.25,0],[0,1]]
mu3 = [0,0]; sigma3 = [[1,0.5],[0.5,1]]
################################################################

################################################################
# 各種パラメータの設定
# 初期点の変更回数
ITERATIONS = 100
# 正規分布のパラメータ選択
MU = mu1
SIGMA = sigma1
# メッシュ数
MESH_NUMBER = 1000
# メッシュの透明度
TRANSPARENCY = 0.9
# メッシュを生成する乱数のSeed設定
SEED_NUMBER = 42
# メッシュのみの図を作るか否か
MAKE_ONLY_MESH = False
################################################################

def main(i,MeshNumber=0,coords_population=None, xx=None, yy=None, ww=None,CreatedMesh = False, mu = None, sigma = None):
    # ディレクトリの指定 実験データ/人口データ/ランダム/２乗/case1
    experimentPathParent = Path(__file__).resolve().parent.parent.parent.parent.parent.joinpath("実験データ/人工データ/メッシュ/正規分布/２乗/case1")
    # 現在の日時を取得
    now = datetime.now()
    # 日時を文字列としてフォーマット
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    # 保存用ディレクトリの指定
    experimentPath = experimentPathParent.joinpath(formatted_now+"_"+str(i+1))
    # 保存用ディレクトリの作成
    os.mkdir(experimentPath) 
    # 結果の保存先
    resultfile = "result_artMesh_Mean_normal_case1.csv"
    with open(experimentPathParent.joinpath(resultfile), "a") as f:
        f.write(formatted_now + "\n")
        f.write(str(i+1)+"回目，np.seedIndex="+str(i)+"\n")
    # 母点の用意
    # 母点の数
    n = 3
    # 母点をランダムに配置する．（初期点）
    np.random.seed(i)
    pnts = 4*np.random.rand (n,2)-2
    # # 確認用の初期点．正しければコメントアウト
    # pnts = np.array([[-1.5,0],[1.5,0],[0,1.4]])
    # 境界（100×100の正方形領域）
    bnd_end = 5
    bnd_poly = Polygon(np.array([[-bnd_end,-bnd_end],[bnd_end,-bnd_end],[bnd_end,bnd_end],[-bnd_end,bnd_end]]))
    # メッシュ点の作成
    # MeshNumber**2の数のメッシュができる．
    MeshNumber = MESH_NUMBER
    if not CreatedMesh:
        coords_population, xx, yy, ww = CreateMesh(-bnd_end,bnd_end,MeshNumber, mu=mu, sigma=sigma)
    with open(experimentPathParent.joinpath(resultfile), "a") as f:
        f.write("メッシュの数:"+ str(MeshNumber**2)+"\n")
    # メッシュデータの描画
    DrawMesh(xx,yy,ww, formatted_now,experimentPath)
    # costの格納
    cost_record = []
    # 初期状態の図示
    vor_polys_box = bounded_voronoi_mult(bnd_poly, pnts)
    draw_voronoi(bnd_poly, pnts, vor_polys_box, coords_population, formatted_now, experimentPath, number = 0)
    # 初期状態のコストを計算
    cost = cost_function(coords_population[:,:2],coords_population[:,2:].ravel(),pnts, non_claster = True, median = False)
    cost_record.append(cost)
    # 初期点の記録
    with open(experimentPathParent.joinpath(resultfile), "a") as f:
        f.write("初期母点\n")
        np.savetxt(f, pnts, fmt = '%f')
    # k-means法
    # ここで最大の繰り返し回数を変更する
    MaxIterations = 100
    # 実行
    optimized_pnts, labels, optimized_cost = weighted_kmeans(coords_population[:,:2],coords_population[:,2:].ravel(), n, pnts = pnts, max_iter = MaxIterations, initial = True, config = True, formatted_now=formatted_now, experimentPath=experimentPath, resultfile = resultfile)
    # 解の描画
    vor_polys_box = bounded_voronoi_mult(bnd_poly, optimized_pnts)
    draw_voronoi(bnd_poly, optimized_pnts, vor_polys_box, coords_population, formatted_now, experimentPath, labels=labels, coloring = True)
    # k-meansの出力のコスト関数値を記録
    cost_record.append(optimized_cost)
    with open(experimentPathParent.joinpath(resultfile), "a") as f:
        f.write("局所最適点\n")
        np.savetxt(f, optimized_pnts, fmt = '%f')
        f.write("optimized cost\n")
        np.savetxt(f, [optimized_cost], fmt = '%f')
    with open(experimentPathParent.joinpath("cost_stock.csv"), "a") as f:
        np.savetxt(f, [optimized_cost], fmt = '%f')
    return 0

def bounded_voronoi_mult(bnd_poly, pnts):
    vor_polys_box = []
    vor_poly_counter_box = []
    # bnds = []
    # 初期状態を図示
    vor_polys, vor_poly_counter_box = bounded_voronoi(bnd_poly, pnts, vor_poly_counter_box)
    for vor_poly in vor_polys:
        vor_polys_box.append(vor_poly)
    # 終わったら削除
    # bnds.append(np.array(bnd_poly.exterior.coords))
    return vor_polys_box #, vor_polys_counter_box

# 有界なボロノイ図を計算する関数


def bounded_voronoi(bnd_poly, pnts, vor_poly_counter_box):
    # 母点がそもそも領域内に含まれているか検証する
    pnts_len = len(pnts)
    # 含まれている母点の保存用
    pnts_included_counter = 0
    vor_counter = 0
    for i in range(pnts_len):
        pnts_judge = Feature(geometry=Point([pnts[i][0], pnts[i][1]]))
        if boolean_point_in_polygon(pnts_judge, Feature(geometry=bnd_poly)):
            pnts_included_counter += 1
    if pnts_included_counter == 0:
        vor_poly_counter_box.append(vor_counter)
        return [list(bnd_poly.exterior.coords[:-1])], vor_poly_counter_box
    elif pnts_included_counter == 1:
        vor_counter += 1
        vor_poly_counter_box.append(vor_counter)
        return [list(bnd_poly.exterior.coords[:-1])], vor_poly_counter_box
    # すべての母点のボロノイ領域を有界にするために，ダミー母点を6個追加
    gn_pnts = np.concatenate([pnts, np.array([[-100, -100], [-100, 300], [50, 600], [50, -600], [300, 300], [300, -100]])])
    # ボロノイ図の計算
    vor = Voronoi(gn_pnts)
    # 各ボロノイ領域をしまうリスト
    vor_polys = []
    # ダミー以外の母点についての繰り返し
    for i in range(len(gn_pnts) - 6):
        vor_counter = 0
        # 閉空間を考慮しないボロノイ領域,ここで各母点が作るボロノイ領域の座標を取得している
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        # 分割する領域をボロノイ領域の共通部分を計算,空白を挟んだxy座標の羅列、一次元のタプルみたいなもの
        i_cells = bnd_poly.intersection(Polygon(vor_poly))
        # 閉空間を考慮したボロノイ領域の頂点座標を格納、座標化したものはタプルのリストになっているのでリスト化？ここの意味はよく分かってない、また、Polygonは座標点を一点だけ重複して数えているのでそこは省いている
        if i_cells.geom_type == "Polygon":
            vor_counter += 1
            vor_polys.append(list(i_cells.exterior.coords[:-1]))
            vor_poly_counter_box.append(vor_counter)
        else:
            for i_cell in i_cells.geoms:
                vor_counter += 1
                vor_polys.append(list(i_cell.exterior.coords[:-1]))
            vor_poly_counter_box.append(vor_counter)
    return vor_polys, vor_poly_counter_box

# ボロノイ図を描画する関数


def draw_voronoi(bnd_poly, pnts, vor_polys_box, coords_population, formatted_now, experimentPath,number=1,labels = None, coloring = False):
    # import mesh
    xmin = pnts[0][0]
    xmax = pnts[0][0]
    ymin = pnts[0][1]
    ymax = pnts[0][1]
    # polygon to numpy
    bnd = np.array(bnd_poly.exterior.coords)
    xmin = np.min(np.array([xmin, np.min(bnd[:, 0])]))
    xmax = np.max(np.array([xmax, np.max(bnd[:, 0])]))
    ymin = np.min(np.array([ymin, np.min(bnd[:, 1])]))
    ymax = np.max(np.array([ymax, np.max(bnd[:, 1])]))
    # ボロノイ図の描画
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    # 母点
    ax.scatter(pnts[:, 0], pnts[:, 1], label="母点", s=10)
    # メッシュ
    np_coords = np.array(coords_population)
    if coloring:
        ax.scatter(np_coords[:, 0], np_coords[:, 1], c=labels, cmap='viridis')
        ax.scatter(pnts[:, 0], pnts[:, 1], c='red', marker='X')
    else:
        ax.scatter(np_coords[:, 0], np_coords[:, 1], label="メッシュ")
    # ボロノイ領域
    poly_vor = PolyCollection(vor_polys_box, edgecolor="black", facecolors="None", linewidth=1.0)
    ax.add_collection(poly_vor)
    # 描画の範囲設定
    ax.set_xlim(xmin-0.01, xmax+0.01)
    ax.set_ylim(ymin-0.01, ymax+0.01)
    ax.set_aspect('equal')
    ax.legend(loc = "upper right")
    if number > 0:
        filename = experimentPath.joinpath("局所最適解ボロノイ図_"+formatted_now+".png")
    else:
        filename = experimentPath.joinpath("初期状態ボロノイ図_"+formatted_now+".png")
    plt.savefig(filename)
    plt.close()

def weighted_kmeans(X, weights, n_clusters, pnts=None, max_iter=100, initial = False, config = False,formatted_now = None, experimentPath = None,resultfile = None):
    # データポイントの数
    n_samples = X.shape[0]
    # ランダムに初期クラスタ中心を選択
    if initial:
        centroids = pnts
    else:
        random_indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[random_indices]

    #　コストの推移の確認のため描画するかしないか場合分け
    if config:
        cost_record = []
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
            cost_record.append(cost_function(X,weights,centroids,labels,non_claster = False,median = False))
        with open(experimentPath.joinpath(resultfile), "a") as f:
            f.write("cost record in iterations\n")
            np.savetxt(f, np.array(cost_record), fmt = '%f')
        draw_cost(cost_record, formatted_now, experimentPath)
    else:
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
        cost_function_value += weights[i]*np.sum((X[i] - cluster_center)**2)
    
    return centroids, labels, cost_function_value

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

# メッシュの生成
def CreateMesh(bndmin, bndmax,N = 200, mu = mu1, sigma = sigma1):
    X = np.linspace(bndmin,bndmax, N)
    Y = np.linspace(bndmin,bndmax, N)
    X, Y = np.meshgrid(X, Y)
    #各点の座標を取得
    points = np.vstack([X.ravel(), Y.ravel()]).T
    # 正規分布の設定
    mean = mu
    cov = sigma
    rv = multivariate_normal(mean, cov)
    # 各点に正規分布の値を格納
    weights = [rv.pdf(point) for point in points]
    # 作ったものを１つに
    coordinates = [np.array([points[i][0], points[i][1], weights[i]]) for i in range(len(points))]
    # MeshGrid仕様に
    weights_grid = np.array(weights).reshape(X.shape)
    return np.array(coordinates), X , Y, weights_grid

# メッシュの描画
def DrawMesh(X_grid, Y_grid, weights_grid, formatted_now = "Now", experimentPath = ""):
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X_grid, Y_grid, weights_grid, cmap="Reds",shading="auto")
    plt.colorbar(label="Weight")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    filename = experimentPath.joinpath("MeshGrid_"+formatted_now+".png")
    plt.savefig(filename)
    # plt.show()
    plt.close()
# コストの描画
def draw_cost(cost_record,formatted_now, experimentPath):
    plt.figure()
    plt.plot(cost_record)
    plt.xlabel("n(回)")
    plt.ylabel("COST")
    filename = experimentPath.joinpath("CostRecord_"+formatted_now+".png")
    plt.savefig(filename)
    plt.close()
    # plt.show()

# メッシュ数の記録
# def draw_mesh_sum(mesh_sum_record,formatted_now, experimentPath):
#     plt.figure()
#     plt.plot(mesh_sum_record)
#     plt.xlabel("n(回)")
#     plt.ylabel("総メッシュ数")
#     filename = experimentPath.joinpath("MeshSumRecord_"+formatted_now+".png")
#     plt.savefig(filename)
#     plt.clf()
#     # plt.show()
    
if __name__ == '__main__':
    coords_population, xx, yy,ww=CreateMesh(bndmin=-5,bndmax=5,N=MESH_NUMBER,mu=MU, sigma=SIGMA)
    for i in range(ITERATIONS):
        main(i,MeshNumber=MESH_NUMBER,coords_population=coords_population, xx=xx, yy=yy, ww=ww,CreatedMesh = True, mu=MU, sigma=SIGMA)
