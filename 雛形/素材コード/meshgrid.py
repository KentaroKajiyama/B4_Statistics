import numpy as np
import random
import matplotlib.pyplot as plt

def CreateMesh(N = 100):
    X = np.linspace(0, 10, N)
    Y = np.linspace(0, 10, N)
    X, Y = np.meshgrid(X, Y)
    #各点の座標を取得
    points = np.vstack([X.ravel(), Y.ravel()]).T
    #各点にランダムな重みをつける
    weights = [random.randint(0, 1000) for _ in range(len(points))]
    # 作ったものを１つに
    coordinates = [np.array([points[i][0], points[i][1], weights[i]]) for i in range(len(points))]
    # MeshGrid仕様に
    weights_grid = np.array(weights).reshape(X.shape)
    return np.array(coordinates), X , Y, weights_grid

def DrawMesh(X_grid, Y_grid, weights_grid, formatted_now = "Now", experimentPath = ""):
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X_grid, Y_grid, weights_grid, shading="auto")
    plt.colorbar(label="Weight")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # filename = experimentPath.joinpath("MeshGrid_"+formatted_now+".png")
    # plt.savefig(filename)
    plt.show()
    
if __name__ == '__main__':
    _, X_grid, Y_grid, weights_grid = CreateMesh(20)
    DrawMesh(X_grid, Y_grid, weights_grid)