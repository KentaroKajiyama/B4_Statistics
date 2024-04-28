# coding: utf-8
import numpy as np
from scipy.optimize import minimize
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from turfpy.measurement import boolean_point_in_polygon
from geojson import Feature, Point
from scipy.spatial.distance import cdist, euclidean
from matplotlib import rcParams
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
    pnts = np.array([[-1.5,0],[1.5,0],[0,1.4]])
    print(pnts)
    #ボロノイ分割する領域
    bnd = np.array([[-5,-5],[5,-5],[5,5],[-5,5]])
    #初期状態を図示
    vor_polys = bounded_voronoi(bnd, pnts)
    draw_voronoi(bnd,pnts,vor_polys)
    #k-means法
    g = np.zeros((n,2))
    eps = 1e-3
    #do while 文を実装
    while 1 :
        for i in range(n):
            g[i][0]= g_function(pnts,i,0)
            g[i][1]= g_function(pnts,i,1)
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
    optimized_vor = bounded_voronoi(bnd, pnts)
    draw_voronoi(bnd,pnts,optimized_vor)
    
    return 0
    
#有界なボロノイ図を計算する関数
def bounded_voronoi(bnd, pnts):
    
    # すべての母点のボロノイ領域を有界にするために，ダミー母点を3個追加
    gn_pnts = np.concatenate([pnts, np.array([[50, 50], [50, -50], [-50, 0]])])

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
def g_function(pnts,i,xy_index):
    postsize = len(pnts)
    #正規分布のパラメーター
    mean = np.array([0,0])
    cov = np.array([[1,0],[0,1]])
    #領域境界の方法をもうすこし工夫したい
    bnd = np.array([[-5,-5],[5,-5],[5,5],[-5,5]])
    vor = bounded_voronoi(bnd, pnts)
    sigma = 0
    counter = 0
    tmp_sigma_upper = 0
    tmp_sigma_lower = 0
    polygon = Feature(geometry = Polygon(vor[i]))
    for j in range(10000):
        sample_point = np.random.multivariate_normal(mean, cov)
        sample_point_judge = Feature(geometry = Point(list(sample_point)))
        if boolean_point_in_polygon(sample_point_judge, polygon):
            tmp_sigma_upper += sample_point[xy_index] 
            counter += 1
    if counter > 0:
        print("counter:",counter)
        sigma += tmp_sigma_upper/counter
    return sigma


if __name__ == '__main__':
    main()
    
    

