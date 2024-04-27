import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# 卒論への path
rootDir = Path(__file__).resolve().parent.parent.parent.parent

df_subset = pd.read_csv(rootDir.joinpath("発表資料/最終発表/ネットワーク最短距離/distance_for_graph.csv"), header=None, names=['X', 'Y'])

# Setting up the figure and axis for the plot
fig, ax = plt.subplots(figsize=(17, 6))

# The indices will be used for bar positions
indices = np.arange(len(df_subset))
set_names = ['荻窪','西荻窪','久我山','富士見ヶ丘']
# Plotting both X and Y values as horizontal bars
ax.barh(indices - 0.1, df_subset['X'], height=0.2, label='直線距離', color='red')
ax.barh(indices + 0.1, df_subset['Y'], height=0.2, label='道路距離', color='blue')

# Adding some plot decorations
ax.set_yticks(indices)
ax.set_yticklabels(set_names, fontsize=25)
ax.set_xticklabels([f"{x}m" for x in ax.get_xticks()], fontsize=25)
ax.legend(fontsize=25)

plt.show()
