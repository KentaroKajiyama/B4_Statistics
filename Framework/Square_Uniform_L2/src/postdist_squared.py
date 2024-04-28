import numpy as np
from scipy.optimize import minimize
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from turfpy.measurement import boolean_point_in_polygon
from geojson import Feature, Point
import sys




"""
問題設定① 
n個のポスト配置、最適な配置は総平均（期待値）で評価する。
----------------------------------------------------------------
仮定
・100×100の正方形領域
・人口分布は一様
・人々はもっとも近いポストに手紙を出す
・距離は直線距離で近似する
----------------------------------------------------------------
プログラムの改善点
・一点一点独立に扱っているので統一性を持たせたい
・挙動を見たいので更新過程も可視化する
・可視化になるべく時間がかからないようにしたい
"""


def main():
    #ポストの用意
    n=2
    pnts = 100*np.random.rand (n,2)
    #ボロノイ分割する領域
    bnd = np.array([[0,0],[100,0],[100,100],[0,100]])
    #初期状態を図示
    vor_polys = bounded_voronoi(bnd, pnts)
    draw_voronoi(bnd,pnts,vor_polys)
    #境界
    bounds = tuple([(0,100) for i in range(n*2)])
    #最適化
    #初期値は一次元配列にする必要がある
    optimized_pnts = minimize(fun=cost_function,x0=pnts.reshape([n*2,]),method='L-BFGS-B',jac=derivative_cost_function,bounds=bounds)
    #解は１次元配列で出てくるので２次元配列に直す必要あり
    optimized_pnts = optimized_pnts.x.reshape([n,2])
    #解の描画
    optimized_vor = bounded_voronoi(bnd, optimized_pnts)
    draw_voronoi(bnd,optimized_pnts,optimized_vor)
    return 0
    
#有界なボロノイ図を計算する関数
def bounded_voronoi(bnd, pnts):
    
    # すべての母点のボロノイ領域を有界にするために，ダミー母点を3個追加
    gn_pnts = np.concatenate([pnts, np.array([[10000, 10000], [10000, -10000], [-10000, 0]])])

    # ボロノイ図の計算
    vor = Voronoi(gn_pnts)

    # 分割する領域をPolygonに
    bnd_poly = Polygon(bnd)

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
def draw_voronoi(bnd,pnts,vor_polys):
    # ボロノイ図の描画
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    # 母点
    ax.scatter(pnts[:,0], pnts[:,1])

    # ボロノイ領域
    poly_vor = PolyCollection(vor_polys, edgecolor="black",facecolors="None", linewidth = 1.0)
    ax.add_collection(poly_vor)

    xmin = np.min(bnd[:,0])
    xmax = np.max(bnd[:,0])
    ymin = np.min(bnd[:,1])
    ymax = np.max(bnd[:,1])

    ax.set_xlim(xmin-0.1, xmax+0.1)
    ax.set_ylim(ymin-0.1, ymax+0.1)
    ax.set_aspect('equal')
    
    plt.show()
    
    # str = input()
    # plt.savefig(str+".png")

#最適化問題をSLSQPで実装する。
#まずは距離関数を定義する

def dist(x,y,px,py):
    return math.sqrt((x-px)**2 + (y-py)**2)

#コスト関数はひとまずモンテカルロ法で近似する
def cost_function(pnts):
    pnts = pnts.reshape([-1,2])
    postsize = len(pnts)
    #領域境界の方法をもうすこし工夫したい
    bnd = np.array([[0,0],[100,0],[100,100],[0,100]])
    vor = bounded_voronoi(bnd, pnts)
    sigma = 0
    for i in range(postsize):
        counter = 0
        tmp_sigma = 0
        polygon = Feature(geometry = Polygon(vor[i]))
        for j in range(100):
            sample_point = 100*np.random.rand(2)
            sample_point_judge = Feature(geometry = Point(list(sample_point)))
            if boolean_point_in_polygon(sample_point_judge, polygon):
                tmp_sigma += dist(sample_point[0], sample_point[1],pnts[i][0],pnts[i][1])
                counter += 1
        if counter > 0:
            sigma += tmp_sigma/counter
    return sigma

#コスト関数の勾配を計算
def derivative_cost_function(pnts):
    pnts = pnts.reshape([-1,2])
    postsize = len(pnts)
    #領域境界の方法をもうすこし工夫したい
    bnd = np.array([[0,0],[100,0],[100,100],[0,100]])
    vor = bounded_voronoi(bnd, pnts)
    sigma = np.zeros((postsize,2))
    for i in range(postsize):
        counter = 0
        tmp_sigma_x = 0
        tmp_sigma_y = 0
        polygon = Feature(geometry= Polygon(vor[i]))
        for j in range(100):
            sample_point = 100*np.random.rand(2)
            sample_point_judge = Feature(geometry=Point(list(sample_point)))
            if boolean_point_in_polygon(sample_point_judge, polygon):
                tmp_sigma_x += (pnts[i][0]-sample_point[0])/dist(sample_point[0], sample_point[1],pnts[i][0],pnts[i][1])
                tmp_sigma_y += (pnts[i][1]-sample_point[1])/dist(sample_point[0], sample_point[1],pnts[i][0],pnts[i][1])
                counter += 1
        if counter > 0:
            sigma[i][0] += tmp_sigma_x/counter
            sigma[i][1] += tmp_sigma_y/counter
    sigma = np.ravel(sigma)
    return sigma

if __name__ == '__main__':
    main()

