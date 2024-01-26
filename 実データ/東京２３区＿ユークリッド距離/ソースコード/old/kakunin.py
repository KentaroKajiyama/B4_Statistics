# coding: utf-8
import geopandas as gpd
from matplotlib import rcParams
import IPython as ip
from shapely.ops import unary_union
rcParams['lines.markersize']= 1.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


gdf_mesh_taito = gpd.read_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/実データ/台東区＿ユークリッド距離/ソースコード/台東区＿メッシュあり.shp")
gdf_mesh_tokyo23 = gpd.read_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/実データ/東京２３区＿ユークリッド距離/ソースコード/メッシュあり東京２３区人口データ付き.shp")
ip.display.display(gdf_mesh_taito.fillna(0))
ip.display.display(gdf_mesh_tokyo23.fillna(0))