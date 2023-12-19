import geopandas as gpd
from shapely.geometry import MultiPoint, Polygon, LineString
from shapely.ops import voronoi_diagram

# ポイントデータのファイルパス
fp_in = '/Users/kajiyamakentarou/Keisu/卒論/最適配置/雛形/素材コード/P20-12_13_GML/P20-12_13.shp'
# 出力ファイルパス
fp_out = '/Users/kajiyamakentarou/Keisu/卒論/最適配置/雛形/素材コード/voronoi_P20-12_13.shp'

# ポイントデータの読込と投影変換
gdf_point = gpd.read_file(fp_in).to_crs(epsg=2444)

# 座標を MultiPoint に格納
points = MultiPoint(
    [(point.x, point.y) for point in gdf_point['geometry']]
)

# ボロノイ分割
diagram = voronoi_diagram(points)

# Extract polygons from the diagram
polygons = []
for geom in diagram.geoms:
    if isinstance(geom, Polygon):
        polygons.append(geom)

# Create GeoDataFrame from the list of Polygons
gdf_voronoi = gpd.GeoDataFrame(geometry=polygons, crs=gdf_point.crs)

# ポイントの属性をポリゴンに付与（空間結合）
gdf_voronoi = gpd.sjoin(gdf_voronoi, gdf_point)

# シェープファイルに出力
gdf_voronoi.to_file(fp_out, encoding='utf-8')
