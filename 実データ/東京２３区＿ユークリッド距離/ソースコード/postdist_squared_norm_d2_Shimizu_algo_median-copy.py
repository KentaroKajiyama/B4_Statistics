# coding: utf-8
import numpy as np
import pandas as pd
import geopandas as gpd
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from shapely.ops import unary_union
from turfpy.measurement import boolean_point_in_polygon
from geojson import Feature, Point
from scipy.spatial.distance import cdist, euclidean
import shp_to_mesh
from pathlib import Path
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
・行政区域は気にしない
・母点がないようなボロノイ領域は一旦無視する
・ボロノイ領域の一つが複数に別れる場合、medianを取ると領域内に収まらない可能性がある。
----------------------------------------------------------------
プログラムの改善点
・一点一点独立に扱っているので統一性を持たせたい
・挙動を見たいので更新過程も可視化する
・可視化になるべく時間がかからないようにしたい
・なぜか45度回転するのでそこのデバッグ＋分布の可視化をやめる
・np.array2dの可視化の際の座標配置は(y,x)の順番になるからそこが関係している可能性はある。
・コスト関数の計算時に明らかに無視できるメッシュがあるのに律儀に読み込んでいるのは計算量の無駄
"""


def main():
    #ポストの用意
    n=23
    # ディレクトリの指定
    parent = Path(__file__).resolve().parent.parent
    # 区役所名を除外して、緯度と経度のみの配列を作成
    df = pd.read_csv(parent.joinpath("初期状態/tokyo_23_wards_offices_utf8.csv"))
    pnts = df[['経度', '緯度']].to_numpy()
    #ボロノイ分割する領域（台東区）bndはPolygon型
    gdf_bound = gpd.read_file(parent.joinpath("ソースコード/tokyo23_polygon.shp"))
    gdf_mesh = gpd.read_file(parent.joinpath("ソースコード/メッシュあり東京２３区人口データ付き.shp")).fillna(0)
    # bnd_polys bnd_polyの複数形
    bnd_polys = unary_union(gdf_bound["geometry"])
    #初期状態を図示
    vor_polys_box, vor_poly_counter_box, bnds = bounded_voronoi_mult(bnd_polys, pnts)
    draw_voronoi(bnd_polys,pnts,vor_polys_box,gdf_mesh)
    #k-means法
    g = np.zeros((n,2))
    eps = 1e-6
    #do while 文を実装
    while 1 :
        vor_poly_counter = 0
        for i in range(n):
            g[i], vor_poly_counter = g_function(pnts, i, bnd_polys, gdf_mesh, vor_poly_counter)
        if norm(g,pnts,eps):
            pnts = g
            break
        #そのままgを渡してしまうと参照渡しとなってしまう？numpy.ndarrayの仕様がわからない
        pnts = np.copy(g)
        print("pnts",pnts)
    #解の描画
    print("optimized points:",pnts)
    optimized_vor_box = cost_function(pnts, bnd_polys, gdf_mesh)
    draw_voronoi(bnd_polys,pnts,optimized_vor_box,gdf_mesh)
    
    return 0

def bounded_voronoi_mult(bnd_polys, pnts):
    vor_polys_box = []
    vor_poly_counter_box = []
    bnds = []
    #初期状態を図示
    for bnd_poly in bnd_polys.geoms:
        vor_polys, vor_poly_counter_box = bounded_voronoi(bnd_poly, pnts, vor_poly_counter_box)
        for vor_poly in vor_polys:
            vor_polys_box.append(vor_poly)
        #終わったら削除
        bnds.append(np.array(bnd_poly.exterior.coords))
    return vor_polys_box, vor_poly_counter_box, bnds

#有界なボロノイ図を計算する関数
def bounded_voronoi(bnd_poly, pnts, vor_poly_counter_box):
    # 母点がそもそも領域内に含まれているか検証する
    pnts_len = len(pnts)
    # 含まれている母点の保存用
    pnts_included_counter = 0
    vor_counter = 0
    for i in range(pnts_len):
        pnts_judge = Feature(geometry = Point([pnts[i][0], pnts[i][1]]))
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
    gn_pnts = np.concatenate([pnts, np.array([[139.3, 35.7],[139.6,36.1],[140.3,35.65],[139.758, 35.35],[140.1,35.55],[140.2,36]])])
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
        if i_cells.geom_type == "Polygon" :
            vor_counter += 1
            vor_polys.append(list(i_cells.exterior.coords[:-1]))
            vor_poly_counter_box.append(vor_counter)
            # gdf = gpd.GeoDataFrame({'geometry': [i_cells]})
            # print("Polygon")
            # print(i_cells)
            # gdf.plot()
            # plt.show()
        else :
            for i_cell in i_cells.geoms :
                vor_counter += 1
                vor_polys.append(list(i_cell.exterior.coords[:-1]))
            vor_poly_counter_box.append(vor_counter)
            # gdf = gpd.GeoDataFrame({'geometry': [i_cell]})
            # print("MultiPolygon")
            # print(i_cell)
            # gdf.plot()
            # plt.show()
    return vor_polys, vor_poly_counter_box

#ボロノイ図を描画する関数
def draw_voronoi(bnd_polys,pnts,vor_polys_box,gdf_mesh):
    # import mesh
    coords_population = shp_to_mesh.shp_to_meshCoords(gdf_mesh)
    xmin = pnts[0][0]
    xmax = pnts[0][0]
    ymin = pnts[0][1]
    ymax = pnts[0][1]
    # polygon to numpy
    for bnd_poly in bnd_polys.geoms:
        bnd = np.array(bnd_poly.exterior.coords)
        xmin = np.min(np.array([xmin, np.min(bnd[:,0])]))
        xmax = np.max(np.array([xmax, np.max(bnd[:,0])]))
        ymin = np.min(np.array([ymin, np.min(bnd[:,1])]))
        ymax = np.max(np.array([ymax, np.max(bnd[:,1])]))
    # ボロノイ図の描画
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    # 母点
    ax.scatter(pnts[:,0], pnts[:,1], label = "母点")
    # メッシュ
    np_coords = np.array(coords_population)
    ax.scatter(np_coords[:,0], np_coords[:,1], label = "メッシュ")
    # ボロノイ領域
    poly_vor = PolyCollection(vor_polys_box, edgecolor="black",facecolors="None", linewidth = 1.0)
    ax.add_collection(poly_vor)
    # 描画の範囲設定
    ax.set_xlim(xmin-0.01, xmax+0.01)
    ax.set_ylim(ymin-0.01, ymax+0.01)
    ax.set_aspect('equal')
    ax.legend()
    plt.show()

#最適化問題をSLSQPで実装する。

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
def g_function(pnts, i, bnd_polys, gdf_mesh, vor_poly_counter):
    #メッシュデータ
    coords_population = shp_to_mesh.shp_to_meshCoords(gdf_mesh)
    #領域境界
    vor_poly_box, vor_poly_counter_box, bnds = bounded_voronoi_mult(bnd_polys, pnts)
    while vor_poly_counter_box[vor_poly_counter] == 0:
        vor_poly_counter += 1
    answer = pnts[i]
    vor_poly_counter_save = vor_poly_counter
    counter = 0
    sample_points = []
    mesh_weights = []
    for vor_poly_num in range(vor_poly_counter_box[vor_poly_counter]):
        polygon = Feature(geometry = Polygon(vor_poly_box[vor_poly_counter_save]))
        for j in range(len(coords_population)):
            sample_point_judge = Feature(geometry = Point([coords_population[j][0], coords_population[j][1]]))
            if boolean_point_in_polygon(sample_point_judge, polygon):
                #ボロノイ領域に入っていればリストにnp.arrayのベクトルを追加
                sample_points.append(np.array([coords_population[j][0], coords_population[j][1]]))
                mesh_weights.append(coords_population[j][2])
                counter += 1
        vor_poly_counter_save += 1
    if counter > 0:
        print("counter:",counter)
        print("length_sample_points:", len(sample_points))
        answer = geometric_median(np.array(sample_points), np.array(mesh_weights))
        if answer[0] == "None":
            answer = pnts[i]
    vor_poly_counter += 1
    return answer, vor_poly_counter

#geometric medianの計算
def geometric_median(X, mesh_weight, eps=1e-5):
    #初期点は平均値から始める
    if X.size == 0:
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

#まずは距離関数を定義する
def dist(x,y,px,py):
    return math.sqrt((x-px)**2 + (y-py)**2)

#コスト関数
def cost_function(pnts, bnd_polys, gdf_mesh):
    sum = 0
    vor_poly_counter = 0
    counter = 0
    #メッシュデータ
    coords_population = shp_to_mesh.shp_to_meshCoords(gdf_mesh)
    #領域境界
    vor_poly_box, vor_poly_counter_box, bnds = bounded_voronoi_mult(bnd_polys, pnts)
    for i in range(len(pnts)):
        tmp_sigma_upper = 0
        tmp_sigma_lower = 0
        while vor_poly_counter < len(vor_poly_counter_box) and vor_poly_counter_box[vor_poly_counter] == 0 :
            vor_poly_counter += 1
        if vor_poly_counter < len(vor_poly_counter_box):
            for vor_poly_num in range(vor_poly_counter_box[vor_poly_counter]):
                polygon = Feature(geometry = Polygon(vor_poly_box[vor_poly_counter]))
                for j in range(len(coords_population)):
                    sample_point_judge = Feature(geometry = Point([coords_population[j][0], coords_population[j][1]]))
                    if boolean_point_in_polygon(sample_point_judge, polygon):
                        #ボロノイ領域に入っていれば和を計算
                        tmp_sigma_upper += coords_population[j][2]*(dist(coords_population[j][0],coords_population[j][1],pnts[i][0],pnts[i][1])**2)
                        tmp_sigma_lower += coords_population[j][2]
                        counter += 1
                vor_poly_counter += 1
        if counter > 0 and tmp_sigma_lower > 0:
            sum += tmp_sigma_upper/tmp_sigma_lower
    print("Optimized Cost:",sum)
    return vor_poly_box


if __name__ == '__main__':
    main()
    

