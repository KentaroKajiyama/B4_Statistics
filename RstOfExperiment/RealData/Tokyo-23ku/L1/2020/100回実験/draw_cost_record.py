import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from matplotlib import rcParams
rcParams['lines.markersize'] = 1.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio','Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def draw_cost_record():
    # 現在の日時を取得
    now = datetime.now()
    # 日時を文字列としてフォーマット
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    file_path = Path(__file__).resolve().parent.joinpath("cost_stock.csv")
    df = pd.read_csv(file_path, header=None)
    data = df.iloc[:,0].values
    # 最小値とそのインデックスを見つける
    min_value = np.min(data)
    min_index = np.argmin(data)
    # データをプロット
    plt.plot(data, label='Data')
    plt.xlabel('初期点の乱数seed')
    plt.ylabel('コスト関数値')

    # 最小値を強調
    plt.scatter(min_index, min_value, color='red', label='Min Value', s=20)

    # 凡例の追加
    plt.legend()

    # 画像の保存
    filename = Path(__file__).resolve().parent.joinpath("CostRecord_"+formatted_now+".png")
    plt.savefig(filename)
    plt.close()
    # グラフの表示
    # plt.show()

if __name__=='__main__':
    draw_cost_record()