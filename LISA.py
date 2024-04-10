import pandas as pd
import math
import os
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn
from matplotlib import colors
import scipy.stats
from scipy import sparse
from libpysal.weights import WSP
from pysal.explore import esda
from pysal.lib import weights
from libpysal.weights import W


class LISAMatrix():

    def __init__(self, contiguity=None, alpha=0.05):
        self.contiguity = contiguity
        self.moran_i_map = None
        self.moran_i_list = None
        self.moran_i = None
        self.lag_zi = None
        self.lag_zj = None
        self.quadrant_map = None
        self.p_norm = None
        self.alpha = alpha
        self.p_norm_map = None
        self.sig_map = None
        self.hot_and_sig_map = None

        if self.contiguity != "queen" and self.contiguity != "rook":
            print("Please specify contiguity: queen or rook?")
            exit(0)

    def fit(self, target_matrix, mask=None):
        if mask is None:
            mask = np.ones(target_matrix.shape)

        w_matrix, v_matrix, y = self.convert_to_normalized_weight_matrix(target_matrix, mask)
        self.moran_i_list, self.moran_i, self.lag_zi, self.lag_zj = self.local_moran_i(w_matrix, v_matrix, y)
        self.moran_i_map = self.convert_moran_i_list_to_map(target_matrix, mask, self.moran_i_list)
        quadrant = self.liza_quadrant(self.lag_zi, self.lag_zj)
        self.quadrant_map = self.convert_moran_i_list_to_map(target_matrix, mask, quadrant)
        self.p_norm_map = self.convert_p_norm_to_map(target_matrix, mask, self.p_norm)
        self.sig_map = np.where(self.p_norm_map < self.alpha, 1, 0)
        self.hot_and_sig_map = np.where((self.quadrant_map == 2) & (self.sig_map == 1), 1, 0)

        return self.moran_i_map, self.quadrant_map, self.moran_i_list, self.moran_i, self.hot_and_sig_map

    def convert_to_normalized_weight_matrix(self, target_matrix, mask_map):

        len_x, len_y = target_matrix.shape[0], target_matrix.shape[1]

        w_matrix = []
        v_matrix = []

        for x1 in range(len_x):
            for y1 in range(len_y):
                if mask_map[x1][y1] == 0:
                    continue

                row = []
                v_row = []
                for x2 in range(len_x):
                    for y2 in range(len_y):
                        if mask_map[x2][y2] == 0:
                            continue

                        if x1 == x2 and y1 == y2:
                            row.append(0)
                        elif self.contiguity == "queen" and self.check_neighbor_queen(x1, y1, x2, y2):
                            row.append(1)
                        elif self.contiguity == "rook" and self.check_neighbor_rook(x1, y1, x2, y2):
                            row.append(1)
                        else:
                            row.append(0)
                        v_row.append(target_matrix[x2][y2])

                w_matrix.append(row)
                v_matrix.append(v_row)
        w_matrix, v_matrix = np.array(w_matrix), np.array(v_matrix)
        w_matrix = preprocessing.normalize(w_matrix, norm="l1")

        y = []

        for x1 in range(len_x):
            for y1 in range(len_y):
                if mask_map[x1][y1] == 0:
                    continue
                y.append(target_matrix[x1][y1])

        return w_matrix, v_matrix, np.array(y)

    def convert_w_matrix_to_neibours(self, w_matrix):
        neibours = {}
        for i in range(w_matrix.shape[0]):
            neibours[i] = []
            for j in range(w_matrix.shape[1]):
                if w_matrix[i][j] != 0:
                    neibours[i].append(j)

        return neibours

    def local_moran_i(self, w_matrix, v_matrix, y):
        N = w_matrix.shape[0]
        print(w_matrix.shape)
        print(v_matrix.shape)
        neibours = self.convert_w_matrix_to_neibours(w_matrix)
        w = W(neibours, silence_warnings=True)

        lisa = esda.moran.Moran_Local(y, w)
        moran_i_list = lisa.Is
        moran_i = np.sum(moran_i_list) / N

        w_lag = weights.spatial_lag.lag_spatial(w, y)

        lag_zi = (y - y.mean()) \
                 / y.std()
        lag_zj = (w_lag - y.mean()) \
                 / y.std()

        self.p_norm = lisa.p_sim

        """
        N = w_matrix.shape[0]
        mean_y = np.sum(v_matrix[0]) / N

        Z_i = np.identity(N) * v_matrix
        Z_i = np.sum(Z_i, axis=1)
        std = np.std(Z_i)
        Z_i = Z_i - mean_y

        Z_j = v_matrix - mean_y
        Z_j = Z_j * w_matrix
        Z_j = np.sum(Z_j, axis=1)

        m2 = np.sum(np.square(Z_i)) / N
        moran_i_list = Z_i * Z_j / m2
        moran_i = np.sum(moran_i_list) / N

        print(moran_i)
        print(moran_i_test)

        lag_zi = Z_i / std
        lag_zj = Z_j / std

        self.p_norm = scipy.stats.norm.sf(abs(Z_i / std))*2
        """

        return moran_i_list, moran_i, lag_zi, lag_zj

    def check_neighbor_queen(self, x1, y1, x2, y2):
        if abs(x1 - x2) > 1 or abs(y1 - y2) > 1:
            return False
        return True

    def check_neighbor_rook(self, x1, y1, x2, y2):
        for new_c in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            new_x = x1 + new_c[0]
            new_y = y1 + new_c[1]
            if new_x == x2 and new_y == y2:
                return True
        return False

    def convert_p_norm_to_map(self, target_matrix, mask_map, p_norm):
        len_x, len_y = target_matrix.shape[0], target_matrix.shape[1]
        out = np.full(target_matrix.shape, np.inf)
        ctr = 0
        for x in range(len_x):
            for y in range(len_y):
                if mask_map[x][y] == 0:
                    continue
                out[x][y] = p_norm[ctr]
                ctr += 1
        return out

    def convert_moran_i_list_to_map(self, target_matrix, mask_map, moran_i_list):
        len_x, len_y = target_matrix.shape[0], target_matrix.shape[1]
        out = np.zeros(target_matrix.shape)
        ctr = 0
        for x in range(len_x):
            for y in range(len_y):
                if mask_map[x][y] == 0:
                    continue
                out[x][y] = moran_i_list[ctr]
                ctr += 1
        return out

    def liza_quadrant(self, lag_zi, lag_zj):
        out = np.zeros(lag_zi.shape)
        for i in range(len(lag_zi)):
            if lag_zi[i] >= 0 and lag_zj[i] >= 0:  # HH
                out[i] = 2
            elif lag_zi[i] >= 0 and lag_zj[i] < 0:  # HL
                out[i] = 1
            elif lag_zi[i] < 0 and lag_zj[i] >= 0:  # LH
                out[i] = -1
            elif lag_zi[i] < 0 and lag_zj[i] < 0:  # LL
                out[i] = -2
        return out

    def plot_moran_i_map(self):
        out = self.moran_i_map
        fig, ax = plt.subplots(figsize=(8, 4))
        ax = sns.heatmap(out.transpose(), robust=False, annot=False, cbar=True, cmap='coolwarm',
                         norm=colors.CenteredNorm())
        ax.invert_yaxis()
        plt.axis("off")
        plt.show()

    def plot_p_norm_map(self):
        out = self.sig_map
        fig, ax = plt.subplots(figsize=(8, 4))
        hmap = colors.ListedColormap(['white', 'black'])
        ax = sns.heatmap(out.transpose(), robust=False, annot=False, cbar=True, cmap=hmap)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['insignificant', "significant"])
        ax.invert_yaxis()
        plt.axis("off")
        plt.show()

    def plot_sig_and_hot(self):
        out = self.hot_and_sig_map
        fig, ax = plt.subplots(figsize=(8, 4))
        hmap = colors.ListedColormap(['white', 'black'])
        ax = sns.heatmap(self.hot_and_sig_map.transpose(), robust=False, annot=False, cbar=True, cmap=hmap)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['insignificant', "significant"])
        ax.invert_yaxis()
        plt.axis("off")
        plt.show()

    def plot_quadrant_map(self):
        out = self.quadrant_map
        fig, ax = plt.subplots(figsize=(8, 4))
        hmap = colors.ListedColormap(['blue', 'lightblue', 'white', 'pink', 'red'])
        ax = sns.heatmap(out.transpose(), robust=False, annot=False, cbar=True, cmap=hmap)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-2, -1, 0, 1, 2])
        cbar.set_ticklabels(['Low-Low', 'Low-High', 'Mask', 'High-Low', "High-High"])
        ax.invert_yaxis()
        plt.axis("off")
        plt.show()

    def scatter_plot_moran_i(self):
        ax = seaborn.kdeplot(self.moran_i_list)
        seaborn.rugplot(self.moran_i_list, ax=ax)
        plt.show()
        f, ax = plt.subplots(1, figsize=(6, 6))
        seaborn.regplot(x=self.lag_zi, y=self.lag_zj, ci=None)
        plt.axvline(0, c='k', alpha=0.5)
        plt.axhline(0, c='k', alpha=0.5)
        # Add text labels for each quadrant
        plt.text(1, 1.5, "HH", fontsize=25)
        plt.text(1, -1.5, "HL", fontsize=25)
        plt.text(-1.5, 1.5, "LH", fontsize=25)
        plt.text(-1.5, -1.5, "LL", fontsize=25)
        plt.xlabel('Z_i', fontsize=15)
        plt.ylabel('Z_j', fontsize=15)
        plt.show()


"""
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

Y = np.load(open('Y_Iowa_2016-2017_7_1.npy', 'rb'))
mask = np.load(open('mask_128_64.npy', 'rb'))

acc_matrix = np.zeros((Y.shape[2], Y.shape[3]))
for b in range(len(Y)):
    acc_matrix += Y[b][0]

#plot_map(convert_to_normalized_weight_matrix(acc_matrix))

LISA = LISAMatrix(contiguity="queen")
moran_i_map, quadrant_map, moran_i_list, moran_i, sig_map = LISA.fit(acc_matrix, mask)
print(moran_i)
LISA.plot_moran_i_map()
LISA.scatter_plot_moran_i()
LISA.plot_quadrant_map()
LISA.plot_p_norm_map()
"""