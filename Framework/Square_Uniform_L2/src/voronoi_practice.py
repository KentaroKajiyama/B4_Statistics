import numpy as np
from scipy.optimize import minimize
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon

def main():
    n=5
    bnd = np.array([[0,0],[100,0],[100,100],[0,100]])
    np.random.seed(0)
    pnts = 100*np.random.rand (n,2)

    #イメージを作る

    #vor = bounded_voronoi(bnd, pnts)
    gn_pnts = np.concatenate([pnts, np.array([[10000, 10000], [10000, -10000], [-10000, 0]])])
    print("gn_pnts:",gn_pnts)
    vor = Voronoi(gn_pnts)
    
    print("vor:",vor)
    bnd_poly = Polygon(bnd)
    print("bnd_poly:",bnd_poly)
    for i in range(len(gn_pnts)-3):
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        i_cell = bnd_poly.intersection(Polygon(vor_poly))
        print("vor_poly",i,":",list(i_cell.exterior.coords[:-1]))
        
    
if __name__ == "__main__":
    main()
    