import geopandas as gpd
import IPython as ip
gdf = gpd.read_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/data/人口分布データ/N03-190101_13_GML_行政区域_東京_h31/N03-19_13_190101.shp")
gdf_dissolved = gdf.dropna(subset=['N03_007']).dissolve(by='N03_007')
gdf_taito = gdf_dissolved[gdf_dissolved["N03_004"]=="台東区"]["geometry"]
gdf_taito.to_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/data/人口分布データ/taito_polygon.shp")