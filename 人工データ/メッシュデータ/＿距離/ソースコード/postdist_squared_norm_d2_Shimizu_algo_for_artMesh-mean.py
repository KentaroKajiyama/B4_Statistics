# coding: utf-8
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.optimize import minimize
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
from datetime import datetime
from matplotlib import rcParams
import meshgrid
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
    # ポストの用意
    n = 10
    # ディレクトリの指定 人工データ/メッシュデータ/
    parent = Path(__file__).resolve().parent.parent.parent
    # ディレクトリの指定 実験データ/人口データ/２乗
    experimentPath = Path(__file__).resolve().parent.parent.parent.parent.parent.joinpath("実験データ/人工データ/２乗")
    # 現在の日時を取得
    now = datetime.now()
    # 日時を文字列としてフォーマット
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    # 結果の保存先
    resultfile = "result_Median_"+formatted_now+".txt"
    with open(experimentPath.joinpath(resultfile), "a") as f:
        f.write(formatted_now + "\n")
    # 区役所名を除外して、緯度と経度のみの配列を作成
    pnts = 100*np.random.rand (n,2)
    # ボロノイ分割する領域（台東区）bndはPolygon型
    coords_population, xx, yy, ww = meshgrid.CreateMesh(20)
    # メッシュデータの描画
    meshgrid.DrawMesh(xx,yy,ww, formatted_now,experimentPath)
    # bnd_polys bnd_polyの複数形
    bnd_polys = Polygon(np.array([[0,0],[100,0],[100,100],[0,100]]))
    # costの格納
    cost_record = []
    # 取り込んだMesh数の記録
    mesh_sum_record = []
    # 初期状態を図示
    vor_polys_box, cost = cost_function(pnts, bnd_polys, coords_population)
    cost_record.append(cost)
    with open(experimentPath.joinpath(resultfile), "a") as f:
        f.write("初期母点\n")
        np.savetxt(f, pnts, fmt = '%f')
        f.write("取り込んだ総メッシュ数:"+str(len(coords_population))+"\n")
    # k-means法
    g = np.zeros((n, 2))
    eps = 1e-6
    # ここで最大の繰り返し回数を変更する
    MaxIterations = 100
    # do while 文を実装
    while_counter = 0
    draw_voronoi(bnd_polys, pnts, vor_polys_box, coords_population,formatted_now, while_counter, experimentPath)
    while 1 :
        vor_poly_counter = 0
        mesh_counter_sum = 0
        mesh_counter_box = []
        for i in range(n):
            g[i], vor_poly_counter, mesh_counter = g_function(pnts, i, bnd_polys, coords_population, vor_poly_counter)
            mesh_counter_sum += mesh_counter
            mesh_counter_box.append(mesh_counter)
        if norm(g, pnts, eps, experimentPath, resultfile):
            pnts = g
            break
        elif while_counter == MaxIterations:
            pnts = np.copy(g)
            break
        # そのままgを渡してしまうと参照渡しとなってしまう？numpy.ndarrayの仕様がわからない
        pnts = np.copy(g)
        while_counter += 1
        vor_polys_box, cost = cost_function(pnts, bnd_polys, coords_population)
        cost_record.append(cost)
        mesh_sum_record.append(mesh_counter_sum)
        with open(experimentPath.joinpath(resultfile), "a") as f:
            f.write(str(while_counter)+"回目の母点\n")
            np.savetxt(f, pnts, fmt = '%f')
            f.write(str(while_counter)+"回目の取り込み総メッシュ数: "+str(mesh_counter_sum)+"\n")
            f.write(str(while_counter)+"回目の取り込み各メッシュ数\n")
            np.savetxt(f, np.array(mesh_counter_box).reshape(1,len(mesh_counter_box)), fmt= "%d")
        draw_voronoi(bnd_polys,pnts,vor_polys_box,coords_population,formatted_now, while_counter, experimentPath)
    #解の描画
    optimized_vor_box, cost = cost_function(pnts, bnd_polys, coords_population)
    cost_record.append(cost)
    mesh_sum_record.append(mesh_counter_sum)
    draw_voronoi(bnd_polys, pnts, optimized_vor_box, coords_population,formatted_now, while_counter, experimentPath)
    draw_cost(cost_record, formatted_now, experimentPath)
    draw_mesh_sum(mesh_sum_record, formatted_now, experimentPath)
    with open(experimentPath.joinpath(resultfile), "a") as f:
            f.write(str(while_counter+1)+"回目の母点or最適点\n")
            np.savetxt(f, pnts, fmt = '%f')
            f.write(str(while_counter+1)+"回目の取り込み総メッシュ数: "+str(mesh_counter_sum)+"\n")
            f.write(str(while_counter+1)+"回目の取り込み各メッシュ数\n")
            np.savetxt(f, np.array(mesh_counter_box).reshape(1,len(mesh_counter_box)), fmt= "%d")
            f.write("\n")
            f.write("cost record\n")
            np.savetxt(f, np.array(cost_record), fmt = '%f')
    return 0

def bounded_voronoi_mult(bnd_polys, pnts):
    vor_polys_box = []
    vor_poly_counter_box = []
    bnds = []
    # 初期状態を図示
    vor_polys, vor_poly_counter_box = bounded_voronoi(
        bnd_polys, pnts, vor_poly_counter_box)
    for vor_poly in vor_polys:
        vor_polys_box.append(vor_poly)
    # 終わったら削除
    bnds.append(np.array(bnd_polys.exterior.coords))
    return vor_polys_box, vor_poly_counter_box, bnds

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


def draw_voronoi(bnd_polys, pnts, vor_polys_box, coords_population, formatted_now, number, experimentPath):
    # import mesh
    xmin = pnts[0][0]
    xmax = pnts[0][0]
    ymin = pnts[0][1]
    ymax = pnts[0][1]
    # polygon to numpy
    bnd = np.array(bnd_polys.exterior.coords)
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
    ax.scatter(np_coords[:, 0], np_coords[:, 1], label="メッシュ")
    # ボロノイ領域
    poly_vor = PolyCollection(
        vor_polys_box, edgecolor="black", facecolors="None", linewidth=1.0)
    ax.add_collection(poly_vor)
    # 描画の範囲設定
    ax.set_xlim(xmin-0.01, xmax+0.01)
    ax.set_ylim(ymin-0.01, ymax+0.01)
    ax.set_aspect('equal')
    ax.legend()
    filename = experimentPath.joinpath(str(number)+"回目ボロノイ図_"+formatted_now+".png")
    plt.savefig(filename)
    plt.close()
    # plt.show()

# 最適化問題をSLSQPで実装する。


# whileループの判定関数
def norm(g, y, eps, experimentPath, resultfile):
    p = len(g)
    n = len(g[0])
    for i in range(p):
        sum = 0
        for j in range(n):
            sum = (g[i][j]-y[i][j])**2
        with open(experimentPath.joinpath(resultfile), "a") as f:
            f.write("第"+str(i+1)+"点の移動距離:"+str(sum)+"\n")
        if sum > eps:
            return 0
    return 1

# コスト関数はモンテカルロ法で近似、正規化定数は1とみなし、標本平均を計算しているだけ。


def g_function(pnts, i, bnd_polys, coords_population, vor_poly_counter):
    # 領域境界
    vor_poly_box, vor_poly_counter_box, bnds = bounded_voronoi_mult(
        bnd_polys, pnts)
    while vor_poly_counter_box[vor_poly_counter] == 0:
        vor_poly_counter += 1
    # python のコピーの仕様がよくわからない．
    answer = np.copy(pnts[i])

    vor_poly_counter_save = vor_poly_counter
    counter = 0
    tmp_sigma_upper_x = 0
    tmp_sigma_upper_y = 0
    tmp_sigma_lower = 0
    for vor_poly_num in range(vor_poly_counter_box[vor_poly_counter]):
        polygon = Feature(geometry=Polygon(
            vor_poly_box[vor_poly_counter_save]))
        for j in range(len(coords_population)):
            sample_point_judge = Feature(geometry=Point(
                [coords_population[j][0], coords_population[j][1]]))
            if boolean_point_in_polygon(sample_point_judge, polygon):
                # ボロノイ領域に入っていればリストにnp.arrayのベクトルを追加
                tmp_sigma_upper_x += coords_population[j][2] * \
                    coords_population[j][0]
                tmp_sigma_upper_y += coords_population[j][2] * \
                    coords_population[j][1]
                tmp_sigma_lower += coords_population[j][2]
                counter += 1
        vor_poly_counter_save += 1
    if counter > 0 and tmp_sigma_lower > 0:
        answer[0] = tmp_sigma_upper_x/tmp_sigma_lower
        answer[1] = tmp_sigma_upper_y/tmp_sigma_lower
    vor_poly_counter += 1
    return answer, vor_poly_counter, counter

# まずは距離関数を定義する


def dist(x, y, px, py):
    return math.sqrt((x-px)**2 + (y-py)**2)

# コスト関数


def cost_function(pnts, bnd_polys, coords_population):
    sum = 0
    vor_poly_counter = 0
    counter = 0
    # 領域境界
    vor_poly_box, vor_poly_counter_box, bnds = bounded_voronoi_mult(bnd_polys, pnts)
    for i in range(len(pnts)):
        tmp_sigma_upper = 0
        tmp_sigma_lower = 0
        while vor_poly_counter < len(vor_poly_counter_box) and vor_poly_counter_box[vor_poly_counter] == 0:
            vor_poly_counter += 1
        if vor_poly_counter < len(vor_poly_counter_box):
            for vor_poly_num in range(vor_poly_counter_box[vor_poly_counter]):
                polygon = Feature(geometry=Polygon(
                    vor_poly_box[vor_poly_counter]))
                for j in range(len(coords_population)):
                    sample_point_judge = Feature(geometry=Point(
                        [coords_population[j][0], coords_population[j][1]]))
                    if boolean_point_in_polygon(sample_point_judge, polygon):
                        # ボロノイ領域に入っていれば和を計算
                        tmp_sigma_upper += coords_population[j][2]*(dist(
                            coords_population[j][0], coords_population[j][1], pnts[i][0], pnts[i][1])**2)
                        tmp_sigma_lower += coords_population[j][2]
                        counter += 1
                vor_poly_counter += 1
        if counter > 0 and tmp_sigma_lower > 0:
            sum += tmp_sigma_upper/tmp_sigma_lower
    return vor_poly_box, sum

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
def draw_mesh_sum(mesh_sum_record,formatted_now, experimentPath):
    plt.figure()
    plt.plot(mesh_sum_record)
    plt.xlabel("n(回)")
    plt.ylabel("総メッシュ数")
    filename = experimentPath.joinpath("MeshSumRecord_"+formatted_now+".png")
    plt.savefig(filename)
    plt.close()
    # plt.show()
    
if __name__ == '__main__':
    main()
