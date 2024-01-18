import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import IPython as ip
gdf = gpd.read_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/実データ/台東区＿ユークリッド距離/ソースコード/台東区＿メッシュあり.shp")
meshcode = gdf["KEY_CODE"]
population = gdf["population"]
ip.display.display(meshcode)
ip.display.display(population)