import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['lines.markersize']= 1.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def bounded_voronoi(bnd, pnts):
    # すべての母点のボロノイ領域を有界にするために，ダミー母点を3個追加
    gn_pnts = np.concatenate([pnts, np.array([[-100,-100], [200,0], [40,200]])])
    # ボロノイ図の計算
    vor = Voronoi(gn_pnts)
    # 各ボロノイ領域をしまうリスト
    vor_polys = []
    # 分割する領域をPolygonに
    bnd_poly = Polygon(bnd)
    # ダミー以外の母点についての繰り返し
    for i in range(len(gn_pnts) - 3):
        # 閉空間を考慮しないボロノイ領域,ここで各母点が作るボロノイ領域の座標を取得している
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        # 分割する領域をボロノイ領域の共通部分を計算,空白を挟んだxy座標の羅列、一次元のタプルみたいなもの
        i_cell = bnd_poly.intersection(Polygon(vor_poly))
        # 閉空間を考慮したボロノイ領域の頂点座標を格納、座標化したものはタプルのリストになっているのでリスト化？ここの意味はよく分かってない、また、Polygonは座標点を一点だけ重複して数えているのでそこは省いている
        vor_polys.append(list(i_cell.exterior.coords[:-1]))
    return vor_polys

def draw_voronoi(bnd,pnts,vor_polys):
    # ボロノイ図の描画
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    # 母点
    ax.scatter(pnts[:,0], pnts[:,1], c='red', s=10, label='母点')

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
    ax.legend()
    plt.show()
    
points = np.random.randint(0, 101, size = (10,2))
bnd = np.array([[0,0],[100,0],[100,100],[0,100]])
vor_polys = bounded_voronoi(bnd, points)
draw_voronoi(bnd, points, vor_polys)

