import pandas as pd
data = pd.read_csv("/Users/kajiyamakentarou/Keisu/卒論/最適配置/data/人口分布データ/tblT001102Q5339_Tokyo_Mesh_250m_欠損値処理済.csv")
data.replace('*',0).fillna(0)
data.replace('*',0).fillna(0).to_csv("/Users/kajiyamakentarou/Keisu/卒論/最適配置/data/人口分布データ/Tokyo_Mesh_250m_欠損値処理済.csv")