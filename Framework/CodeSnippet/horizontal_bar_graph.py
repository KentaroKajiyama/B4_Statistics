import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
# 水平棒グラフで2つの値を比較
#保存先
savepath = Path(__file__).resolve().parent.parent.parent.parent.joinpath("執筆/figure/secAnalysis")

plt.figure(figsize=(8,3.8))
# 比較する2つの値
l1values2000 = [154175.749013, 206981.95134363353]
l1values2020 = [179574.804532, 237082.74913310324]
l2values2000 = [3335.958899,6923.392824764322]
l2values2020 = [3811.575954,7755.701892948738]
l1labels = ['$L_1$最適解', '区役所']
l2labels = ['$L_2$最適解', '区役所']
l1color2000 = ['purple', 'green']
l1color2020 = ['deepskyblue', 'green']
l2color2000 = ['orange', 'green']
l2color2020 = ['blue', 'green']
# 水平棒グラフを作成
plt.barh(l2labels, l2values2000, color=l2color2000)

# タイトルと軸ラベルを追加
plt.xlabel('Cost')
# ファイル名の入力
l1filename2000 = "2000_L1_Cost.png"
l1filename2020 = "2020_L1_Cost.png"
l2filename2000 = "2000_L2_Cost.png"
l2filename2020 = "2020_L2_Cost.png"
# グラフを表示
plt.savefig(savepath.joinpath(l2filename2000))
