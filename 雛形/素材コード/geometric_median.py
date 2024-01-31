import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm
from scipy.spatial.distance import cdist, euclidean
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


#input ndarray(2*n), output ndarray(2*1)
def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

# visualization
rng = np.random.default_rng()
inputs = rng.random((10,2))
median = geometric_median(inputs)
mean = np.mean(inputs,axis=0)
lines = [[(inputs[i][0],inputs[i][1]), (median[0],median[1])] for i in range(inputs.shape[0])]
lc = mc.LineCollection(lines, linewidth = 1, color = (0,0,0,0.2))
fig, ax = plt.subplots()
ax.scatter(inputs[:, 0], inputs[:, 1],color='blue', label='初期点')
ax.scatter(median[0],median[1],color='red',s= 100, label='$L_1$-中央値')
ax.scatter(mean[0],mean[1],color='green',s=80, label='平均値')
ax.add_collection(lc)
ax.legend()
plt.show()