# coding: utf-8
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import IPython as ip

def get_latlon(meshCode):
    # Meshの左端の座標を計算している
    # 文字列に変換
    meshCode = str(meshCode)

    # １次メッシュ用計算
    code_first_two = meshCode[0:2]
    code_last_two = meshCode[2:4]
    code_first_two = int(code_first_two)
    code_last_two = int(code_last_two)
    lat  = code_first_two * 2 / 3
    lon = code_last_two + 100

    if len(meshCode) > 4:
        # ２次メッシュ用計算
        if len(meshCode) >= 6:
            code_fifth = meshCode[4:5]
            code_sixth = meshCode[5:6]
            code_fifth = int(code_fifth)
            code_sixth = int(code_sixth)
            lat += code_fifth * 2 / 3 / 8
            lon += code_sixth / 8

        # ３次メッシュ用計算
        if len(meshCode) >= 8:
            code_seventh = meshCode[6:7]
            code_eighth = meshCode[7:8]
            code_seventh = int(code_seventh)
            code_eighth = int(code_eighth)
            lat += code_seventh * 2 / 3 / 8 / 10
            lon += code_eighth / 8 / 10
        
        # 4次メッシュ用計算
        if len(meshCode) >= 9:
            code_nineth = meshCode[8:9]
            code_nineth = int(code_nineth)
            if code_nineth > 2:
                lat += 1 * 2 / 3 / 8 / 10 / 2
            if code_nineth % 2 == 0:   
                lon += 1 / 8 / 10 / 2
        
        # 5次メッシュ用計算
        if len(meshCode) >= 10:
            code_tenth = meshCode[9:10]
            code_tenth = int(code_tenth)
            if code_tenth > 2:
                lat += 1 * 2 / 3 / 8 / 10 / 2 / 2
            if code_tenth % 2 == 0:   
                lon += 1 / 8 / 10 / 2 / 2
        
        # 6次メッシュ用計算
        if len(meshCode) == 11:
            code_eleventh = meshCode[10:11]
            code_eleventh = int(code_eleventh)
            if code_eleventh > 2:
                lat += 1 * 2 / 3 / 8 / 10 / 2 / 2 / 2
            if code_eleventh % 2 == 0:
                lon += 1 / 8 / 10 / 2 / 2 / 2
    # 点をメッシュの中央に置く
    if len(meshCode) == 10:
        lat += 1 * 2 / 3 / 8 / 10 / 2 / 2 / 2
        lon += 1 / 8 / 10 / 2 / 2 / 2
    
    return lon, lat

def shp_to_meshCoords(gdf):
    meshCodeSeries = gdf["KEY_CODE"]
    populationSeries = gdf["population"]
    length  = len(meshCodeSeries)
    coords_populations = []
    for i in range(length):
        if ~np.isnan(populationSeries[i]):
            lon, lat = get_latlon(meshCodeSeries[i])
            tmp = [lon, lat, populationSeries[i]]
            coords_populations.append(tmp)
    return coords_populations

if __name__ == '__main__':
    gdf = gpd.read_file("/Users/kajiyamakentarou/Keisu/卒論/最適配置/実データ/台東区＿ユークリッド距離/ソースコード/台東区＿メッシュあり.shp")
    coords_populations = shp_to_meshCoords(gdf)
    print(coords_populations)