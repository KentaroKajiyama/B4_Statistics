import numpy as np
from scipy.optimize import minimize
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from turfpy.measurement import boolean_point_in_polygon
from geojson import Feature, Point

def main():
    print("input path of text file")
    s = input()
    pnts = read_text_file(s)
    #ボロノイ分割する領域
    bnd = np.array([[-5,-5],[5,-5],[5,5],[-5,5]])
    vor_polys = bounded_voronoi(bnd, pnts)
    draw_voronoi(bnd,pnts,vor_polys)
    
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
    
#数値ファイルを読み取るプログラム
def read_text_file(file_path):
    # ファイルを開いて中の内容を読み込む
    with open(file_path, 'r') as file:
        # 各行のデータをリストに格納
        data_lines = file.readlines()

    # 各行のデータを数値に変換し、2次元配列に格納
    numeric_data = [list(map(float, line.split())) for line in data_lines]

    # NumPyの2次元配列に変換
    numpy_array = np.array(numeric_data)

    return numpy_array


if __name__ == '__main__':
    main()