# coding: utf-8
import numpy as np
import geopandas as gpd
from scipy.optimize import minimize
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from turfpy.measurement import boolean_point_in_polygon
from geojson import Feature, Point
from scipy.spatial.distance import cdist, euclidean
import shp_to_mesh
from matplotlib import rcParams
rcParams['lines.markersize']= 1.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

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
・サンプル点は互いに被らないバラバラなものとする
----------------------------------------------------------------
プログラムの改善点
・一点一点独立に扱っているので統一性を持たせたい
・挙動を見たいので更新過程も可視化する
・可視化になるべく時間がかからないようにしたい
・なぜか45度回転するのでそこのデバッグ＋分布の可視化をやめる
・np.array2dの可視化の際の座標配置は(y,x)の順番になるからそこが関係している可能性はある。
"""


def main():
    #ポストの用意
    n=3
    pnts = np.array([[139.77289, 35.72038],[139.7933,35.72189],[139.78465,35.70103]])
    #ボロノイ分割する領域（台東区）bndはPolygon型
    gdf_bound = gpd.read_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/data/人口分布データ/taito_polygon.shp")
    gdf_mesh = gpd.read_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/実データ/台東区＿ユークリッド距離/ソースコード/台東区＿メッシュあり.shp")
    print(gdf_mesh.dropna)
    bnd_poly = gdf_bound["geometry"].iloc[0]
    #初期状態を図示
    vor_polys = bounded_voronoi(bnd_poly, pnts)
    draw_voronoi(bnd_poly,pnts,vor_polys,gdf_mesh)
    #k-means法
    g = np.zeros((n,2))
    eps = 1e-6
    #do while 文を実装
    while 1 :
        for i in range(n):
            g[i] = g_function(pnts, i, bnd_poly, gdf_mesh)
        if norm(g,pnts,eps):
            pnts = g
            break
        #そのままgを渡してしまうと参照渡しとなってしまう？numpy.ndarrayの仕様がわからない
        pnts = np.copy(g)
        print("pnts",pnts)
    #解の描画
    print("optimized points:",pnts)
    optimized_vor = bounded_voronoi(bnd_poly, pnts)
    draw_voronoi(bnd_poly,pnts,optimized_vor,gdf_mesh)
    
    return 0
    
#有界なボロノイ図を計算する関数
def bounded_voronoi(bnd_poly, pnts):
    
    # すべての母点のボロノイ領域を有界にするために，ダミー母点を3個追加
    gn_pnts = np.concatenate([pnts, np.array([[139.84,35.8], [139.9,35.6], [139.6,35.695]])])

    # ボロノイ図の計算
    vor = Voronoi(gn_pnts)

    # 分割する領域をPolygonに
    # bnd_poly = Polygon(bnd)

    # 各ボロノイ領域をしまうリスト
    vor_polys = []

    # ダミー以外の母点についての繰り返し
    for i in range(len(gn_pnts) - 3):

        # 閉空間を考慮しないボロノイ領域,ここで各母点が作るボロノイ領域の座標を取得している
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        # 分割する領域をボロノイ領域の共通部分を計算,空白を挟んだxy座標の羅列、一次元のタプルみたいなもの
        i_cell = bnd_poly.intersection(Polygon(vor_poly))

        # 閉空間を考慮したボロノイ領域の頂点座標を格納、座標化したものはタプルのリストになっているのでリスト化？ここの意味はよく分かってない、また、Polygonは座標点を一点だけ重複して数えているのでそこは省いている
        vor_polys.append(list(i_cell.exterior.coords[:-1]))
    return vor_polys

#ボロノイ図を描画する関数
def draw_voronoi(bnd_poly,pnts,vor_polys,gdf_mesh):
    # import mesh
    coords_population = shp_to_mesh.shp_to_meshCoords(gdf_mesh)
    # polygon to numpy
    bnd = np.array(bnd_poly.exterior.coords)
    # ボロノイ図の描画
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    # 母点
    ax.scatter(pnts[:,0], pnts[:,1], label = "母点")
    # メッシュ
    np_coords = np.array(coords_population)
    ax.scatter(np_coords[:,0], np_coords[:,1], label = "メッシュ")
    # ボロノイ領域
    poly_vor = PolyCollection(vor_polys, edgecolor="black",facecolors="None", linewidth = 1.0)
    ax.add_collection(poly_vor)

    xmin = np.min(bnd[:,0])
    xmax = np.max(bnd[:,0])
    ymin = np.min(bnd[:,1])
    ymax = np.max(bnd[:,1])

    ax.set_xlim(xmin-0.01, xmax+0.01)
    ax.set_ylim(ymin-0.01, ymax+0.01)
    ax.set_aspect('equal')
    ax.legend()
    plt.show()
    
    # str = input()
    # plt.savefig(str+".png")

#最適化問題をSLSQPで実装する。

#まずは距離関数を定義する
def dist(x,y,px,py):
    return math.sqrt((x-px)**2 + (y-py)**2)

#whileループの判定関数
def norm(g,y,eps):
    print("g",g)
    print("y",y)
    print("eps",eps)
    p = len(g)
    n = len(g[0])
    for i in range(p):
        sum = 0
        for j in range(n):
            sum += (g[i][j]-y[i][j])**2
            print("sum",sum)
        if sum > eps:
            return 0
    return 1

#コスト関数はモンテカルロ法で近似、正規化定数は1とみなし、標本平均を計算しているだけ。
def g_function(pnts, i, bnd_poly, gdf_mesh):
    #メッシュデータ
    coords_population = shp_to_mesh.shp_to_meshCoords(gdf_mesh)
    #領域境界
    vor = bounded_voronoi(bnd_poly, pnts)
    answer = pnts[i]
    counter = 0
    polygon = Feature(geometry = Polygon(vor[i]))
    sample_points = []
    mesh_weights = []
    for j in range(len(coords_population)):
        sample_point_judge = Feature(geometry = Point([coords_population[j][0], coords_population[j][1]]))
        if boolean_point_in_polygon(sample_point_judge, polygon):
            #ボロノイ領域に入っていればリストにnp.arrayのベクトルを追加
            sample_points.append(np.array([coords_population[j][0], coords_population[j][1]]))
            mesh_weights.append(coords_population[j][2])
            counter += 1
    if counter > 0:
        print("counter:",counter)
        answer = geometric_median(np.array(sample_points), np.array(mesh_weights))
    return answer

#geometric medianの計算
def geometric_median(X, mesh_weight, eps=1e-5):
    #初期点は平均値から始める
    y = np.mean(X, 0)
    mesh_weight = mesh_weight.reshape([-1,1])
    print(mesh_weight.shape)
    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]
        zero = (D == 0)[:, 0]
        Dinv = mesh_weight[nonzeros] / D[nonzeros]
        Dinvs = np.sum(Dinv)
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
        # 閾値を超えた時に終了
        if euclidean(y, y1) < eps:
            return y1

        y = y1

if __name__ == '__main__':
    main()
    
    

