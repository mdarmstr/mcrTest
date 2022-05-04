# This is a sample Python script.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pymcr.mcr import McrAR

from pymcr.constraints import ConstraintNonneg, ConstraintNorm


def mcrTest(D, R):
    # X = CS.T
    sz = np.shape(D)
    # Initialisation of the component C
    C = np.random.rand(sz[0], R)
    S = np.random.rand(sz[1], R)
    ssr2 = 1e9
    ssr1 = np.linalg.norm(D - C @ S.T) ** 2
    eps = 1e-12

    while abs(ssr1 - ssr2) / ssr2 > eps:
        ssr1 = ssr2
        C = D @ S @ np.linalg.pinv(S.T @ S)
        S = D.T @ C @ np.linalg.pinv(C.T @ C)

        ssr2 = np.linalg.norm(D - C @ S.T) ** 2
        print(ssr2)

    return C, S




if __name__ == '__main__':
    D = pd.read_csv("gcms1.csv", sep=",").to_numpy()
    C, S = mcrTest(D, 3)
    plt.plot(C)
    plt.show()


