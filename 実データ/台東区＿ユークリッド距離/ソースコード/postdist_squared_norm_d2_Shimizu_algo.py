# coding: utf-8
import numpy as np
import geopandas as gpd
import math
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from turfpy.measurement import boolean_point_in_polygon
from geojson import Feature, Point
import shp_to_mesh
from matplotlib import rcParams
rcParams['lines.markersize']=1.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

"""
問題設定① 
n個のポスト配置、最適な配置は総平均（期待値）で評価する。
東京２３区で実験
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
    #ポストの用意
    n=3
    pnts = np.array([[139.77289, 35.72038],[139.7933,35.72189],[139.78465,35.70103]])
    #ボロノイ分割する領域（東京２３区）bndはPolygon型
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
            g[i][0]= g_function(pnts,i,0,bnd_poly, gdf_mesh)
            g[i][1]= g_function(pnts,i,1,bnd_poly, gdf_mesh)
        print("g",g)
        if norm(g,pnts,eps):
            pnts = g
            break
        #そのままgを渡してしまうと参照渡しとなってしまう？numpy.ndarrayの仕様がわからない
        pnts = np.copy(g)
        print("pnts",pnts)
    #解の描画
    print("optimized points:",pnts)
    #print("cost:",cost_function(pnts))
    optimized_vor = bounded_voronoi(bnd_poly, pnts)
    draw_voronoi(bnd_poly,pnts,optimized_vor,gdf_mesh)
    
    return 0
    
#有界なボロノイ図を計算する関数
def bounded_voronoi(bnd_poly, pnts):
    
    # すべての母点のボロノイ領域を有界にするために，ダミー母点を3個追加
    gn_pnts = np.concatenate([pnts, np.array([[139.84,35.8], [139.9,35.6], [139.6,35.695]])])

    # ボロノイ図の計算
    vor = Voronoi(gn_pnts)
    
    # print("vor.points:",vor.points)
    # print("vor.vertices:",vor.vertices)
    # print("vor.ridge_points:",vor.ridge_points)
    # print("vor.ridge_vertices:",vor.ridge_vertices)
    # print("vor.regions:",vor.regions)
    # print("vor.point_region:",vor.point_region)
    # print("vor.furthest_site:",vor.furthest_site)
    
    # fig, ax = plt.subplots()
    # # 分割する領域がnumpy.arrayならコメントアウトを解除してPolygonに
    # # bnd_poly = Polygon(bnd)
    # bnd_x, bnd_y = bnd_poly.exterior.xy
    # ax.plot(bnd_x,bnd_y)
    # voronoi_plot_2d(vor,ax=ax)
    # plt.show()
    # 各ボロノイ領域をしまうリスト
    vor_polys = []

    # ダミー以外の母点についての繰り返し
    for i in range(len(gn_pnts) - 3):

        # 閉空間を考慮しないボロノイ領域,ここで各母点が作るボロノイ領域の座標を取得している
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        # print(vor_poly)
        # 分割する領域をボロノイ領域の共通部分を計算,空白を挟んだxy座標の羅列、一次元のタプルみたいなもの
        i_cell = bnd_poly.intersection(Polygon(vor_poly))
        # print(i_cell)
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
def g_function(pnts,i,xy_index,bnd_poly, gdf_mesh):
    #メッシュデータ
    coords_population = shp_to_mesh.shp_to_meshCoords(gdf_mesh)
    #領域境界
    vor = bounded_voronoi(bnd_poly, pnts)
    sigma = 0
    counter = 0
    tmp_sigma_upper = 0
    tmp_sigma_lower = 0
    polygon = Feature(geometry = Polygon(vor[i]))
    for j in range(len(coords_population)):
        sample_point_judge = Feature(geometry = Point([coords_population[j][0], coords_population[j][1]]))
        if boolean_point_in_polygon(sample_point_judge, polygon):
            tmp_sigma_upper += coords_population[j][2]*coords_population[j][xy_index]
            tmp_sigma_lower += coords_population[j][2]
            counter += 1
    if counter > 0:
        print("counter:",counter)
        sigma += tmp_sigma_upper/tmp_sigma_lower
    return sigma


if __name__ == '__main__':
    main()
    
    

