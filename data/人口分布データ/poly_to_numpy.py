from shapely.geometry import Polygon
import numpy as np

# 例として Polygon オブジェクトを作成
polygon = Polygon([(0, 0), (1, 1), (1, 0)])

# Polygon の外側の境界座標を取得
exterior_coords = np.array(polygon.exterior.coords)

print(exterior_coords)
