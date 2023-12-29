import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
gdf = gpd.read_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/data/人口分布データ/taito_polygon.shp")
polygon_df = gdf["geometry"].iloc[0]
coords = np.asarray(polygon_df)
#polygon = Polygon(coords)
print(coords)