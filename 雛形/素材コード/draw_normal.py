import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# 正規分布のパラメータ
mu1 = [0,0]; sigma1 = [[1,0],[0,1]]
mu2 = [0,0]; sigma2= [[2.25,0],[0,1]]
mu3 = [0,0]; sigma3 = [[1,0.5],[0.5,1]]
mu4 = [0,0]; sigma4= [[9,0],[0,1]]

# 平均と分散共分散行列の定義
mean = mu1
cov_matrix = sigma1

# 正規分布を生成するためのサンプルデータ
num_samples = 1000
samples = np.random.multivariate_normal(mean, cov_matrix, num_samples)

# 楕円を描画する関数
def plot_ellipse(ax, mean, cov_matrix, color='blue', label= None):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # 楕円のパラメータ
    width, height = 2 * np.sqrt(eigenvalues)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, color=color, alpha=0.3, fill= False)

    # 楕円を描画
    ax.add_patch(ell)
    # 楕円の境界線にラベルを設定
    line = Line2D([0], [0], color=color, linestyle='solid', label=label)
    ax.add_line(line)


# geometric median & mean
# sample = np.array([[-1.5,0],[1.5,0],[0,1.4]])
# geometric_median = np.array([[-1.38580159,-0.32264269],[ 1.31299022,-0.35067516],[-0.01921637,0.71046867]])
geometric_mean = np.array([[-1.96,0],[1.96,0],[0,-0.91],[0,0.91]])
# 描画
plot_ellipse(plt.gca(), mean, cov_matrix, label='正規分布')
# plt.scatter(sample[:,0], sample[:,1],color ="blue",label = "初期点")
plt.scatter(geometric_mean[:,0], geometric_mean[:,1], color="red",label="主要点", marker="x")
# plt.scatter(geometric_median[:,0], geometric_median[:,1], color="red",label="本研究")

# 軸の範囲を設定
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# グラフの表示
plt.legend()
plt.show()
