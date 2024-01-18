import geopandas as gpd
import IPython as ip
#shapeファイルの読み込み
gdf = gpd.read_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/data/人口分布データ/N03-190101_13_GML_行政区域_東京_h31/N03-19_13_190101.shp")
#このdissolveで行政区域ごとに一つにまとめる'N03_007'は行政区域コード
gdf_dissolved = gdf.dropna(subset=['N03_007']).dissolve(by='N03_007')
# dissolvedの中から東京２３区の座標データ（ポリゴンデータ）だけ取り出している
gdf_23 = gdf_dissolved.iloc[0:23]["geometry"]
gdf_23.to_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/data/２３区作成用/tokyo23_polygon.shp")