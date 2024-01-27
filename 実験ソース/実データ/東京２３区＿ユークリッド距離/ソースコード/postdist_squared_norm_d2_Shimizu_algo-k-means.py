# coding: utf-8
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import math
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from shapely.ops import unary_union
from turfpy.measurement import boolean_point_in_polygon
from geojson import Feature, Point
from scipy.spatial.distance import cdist, euclidean
import shp_to_mesh
from pathlib import Path
from datetime import datetime
from matplotlib import rcParams
rcParams['lines.markersize'] = 1.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio','Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

"""
問題設定① 
n個のポスト配置、最適な配置は総平均（期待値）で評価する。
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


def main():
    # ディレクトリの指定 東京２３区ユークリッド距離
    parent = Path(__file__).resolve().parent.parent
    # ディレクトリの指定 実験データ東京２３区２乗
    experimentPath = Path(__file__).resolve().parent.parent.parent.parent.joinpath("実験データ/東京２３区/２乗")
    # 現在の日時を取得
    now = datetime.now()
    # 日時を文字列としてフォーマット
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    # 保存用ディレクトリの指定
    experimentPath = experimentPath.joinpath(formatted_now)
    # 保存用ディレクトリの作成
    os.mkdir(experimentPath) 
    # 結果の保存先
    resultfile = "result_Mean_"+formatted_now+".csv"
    with open(experimentPath.joinpath(resultfile), "a") as f:
        f.write(formatted_now + "\n")
    # 母点の用意， 対象領域の用意
    # 母点の数
    n = 23
    # 区役所名を除外して、緯度と経度のみの配列を作成．これをまずは初期点とする．
    df = pd.read_csv(parent.joinpath("初期状態/tokyo_23_wards_offices_utf8.csv"))
    pnts = df[['経度', '緯度']].to_numpy()
    # ボロノイ分割する領域（東京23区）bndはPolygon型
    gdf_bound = gpd.read_file(parent.joinpath("ソースコード/tokyo23_polygon.shp"))
    gdf_mesh_origin = gpd.read_file(parent.joinpath("ソースコード/メッシュあり東京２３区人口データ付き.shp")).fillna(0)
    coords_population = np.array(shp_to_mesh.shp_to_meshCoords(gdf_mesh_origin))
    # bnd_polys bnd_polyの複数形
    bnd_polys = unary_union(gdf_bound["geometry"])
    # costの格納
    cost_record = []
    # 初期状態の図示
    vor_polys_box = bounded_voronoi_mult(bnd_polys, pnts)
    draw_voronoi(bnd_polys, pnts, vor_polys_box, coords_population, formatted_now, experimentPath, number = 0)
    # 初期状態のコストを計算
    cost = cost_function(coords_population[:,:2],coords_population[:,2:].ravel(),pnts, non_claster = True, median = False)
    cost_record.append(cost)
    # 初期点の記録
    with open(experimentPath.joinpath(resultfile), "a") as f:
        f.write("初期母点\n")
        np.savetxt(f, pnts, fmt = '%f')
    # k-means法
    # ここで最大の繰り返し回数を変更する
    MaxIterations = 100
    # 実行
    optimized_pnts, labels, cost = weighted_kmeans(coords_population[:,:2],coords_population[:,2:].ravel(), n, pnts = pnts, max_iter = MaxIterations, initial = True, config = True, formatted_now=formatted_now, experimentPath=experimentPath, resultfile = resultfile)
    # 解の描画
    vor_polys_box = bounded_voronoi_mult(bnd_polys, optimized_pnts)
    draw_voronoi(bnd_polys, optimized_pnts, vor_polys_box, coords_population, formatted_now, experimentPath, labels=labels, coloring = True)
    # k-meansの出力のコスト関数値を記録
    cost_record.append(cost)
    with open(experimentPath.joinpath(resultfile), "a") as f:
            f.write("局所最適点\n")
            np.savetxt(f, optimized_pnts, fmt = '%f')
            f.write("cost record\n")
            np.savetxt(f, np.array(cost_record), fmt = '%f')
    return 0

def bounded_voronoi_mult(bnd_polys, pnts):
    vor_polys_box = []
    vor_poly_counter_box = []
    # bnds = []
    # 初期状態を図示
    for bnd_poly in bnd_polys.geoms:
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
    # すべての母点のボロノイ領域を有界にするために，ダミー母点を3個追加
    gn_pnts = np.concatenate([pnts, np.array([[139.3, 35.], [139.6, 36.1], [140.3, 35.65], [139.758, 35.35], [140.1, 35.55], [140.2, 36]])])
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


def draw_voronoi(bnd_polys, pnts, vor_polys_box, coords_population, formatted_now, experimentPath,number=1,labels = None, coloring = False):
    # import mesh
    xmin = pnts[0][0]
    xmax = pnts[0][0]
    ymin = pnts[0][1]
    ymax = pnts[0][1]
    # polygon to numpy
    for bnd_poly in bnd_polys.geoms:
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
    ax.legend()
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
    main()
