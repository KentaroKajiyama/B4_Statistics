import numpy as np
import pandas as pd
from pathlib import Path

# current directory
current_dir = Path(__file__).resolve().parent
# 母点の用意
df = pd.read_csv(current_dir.joinpath(filename), header=None)
string_array = df.to_numpy()
# 文字列を空白で分割し、浮動小数点数に変換して2次元配列を作成
pnts = np.array([list(map(float, row[0].split())) for row in string_array])

kokyo = np.array([35.686,139.753])
sum_2d = np.sum((pnts-kokyo)**2)
sum_1d = np.sum(np.abs(pnts-kokyo))
max = np.max(np.abs(pnts-kokyo))
print("sum_1d: ", sum_1d);print("sum_2d: ", sum_2d);print("max: ", max)
