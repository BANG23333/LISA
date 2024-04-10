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
from libpysal.weights import WSP

class GearyMatrix():

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

        neibours = self.convert_w_matrix_to_neibours(w_matrix)
        w = W(neibours, silence_warnings=True)
        
        print(w_matrix.shape)
        print(v_matrix.shape)
        
        # print(w)
        lisa = Geary_Local(w)
        # print(y.shape)
        lisa = lisa.fit(y)
        moran_i_list = lisa.localG
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


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import esda
from esda.crand import _prepare_univariate
from esda.crand import crand as _crand_plus
from esda.crand import njit as _njit




class Geary_Local(BaseEstimator):

    """Local Geary - Univariate"""



    def __init__(
        self,
        connectivity=None,
        labels=False,
        sig=0.05,
        permutations=999,
        n_jobs=1,
        keep_simulations=True,
        seed=None,
        island_weight=0,
        drop_islands=False,
    ):
        """
        Initialize a Local_Geary estimator

        Parameters
        ----------
        connectivity     : scipy.sparse matrix object
                           the connectivity structure describing
                           the relationships between observed units.
                           Need not be row-standardized.
        labels           : boolean
                           (default=False)
                           If True use, label if an observation
                           belongs to an outlier, cluster, other,
                           or non-significant group. 1 = outlier,
                           2 = cluster, 3 = other, 4 = non-significant.
                           Note that this is not the exact same as the
                           cluster map produced by GeoDa.
        sig              : float
                           (default=0.05)
                           Default significance threshold used for
                           creation of labels groups.
        permutations     : int
                           (default=999)
                           number of random permutations for calculation
                           of pseudo p_values
        n_jobs           : int
                           (default=1)
                           Number of cores to be used in the conditional
                           randomisation. If -1, all available cores are used.
        keep_simulations : Boolean
                           (default=True)
                           If True, the entire matrix of replications under
                           the null is stored in memory and accessible;
                           otherwise, replications are not saved
        seed             : None/int
                           Seed to ensure reproducibility of conditional
                           randomizations. Must be set here, and not outside
                           of the function, since numba does not correctly
                           interpret external seeds nor
                           numpy.random.RandomState instances.
        island_weight :
            value to use as a weight for the "fake" neighbor for every island.
            If numpy.nan, will propagate to the final local statistic depending
            on the `stat_func`. If 0, then the lag is always zero for islands.
        drop_islands : bool (default True)
            Whether or not to preserve islands as entries in the adjacency
            list. By default, observations with no neighbors do not appear
            in the adjacency list. If islands are kept, they are coded as
            self-neighbors with zero weight. See ``libpysal.weights.to_adjlist()``.

        Attributes
        ----------
        localG          : numpy array
                          array containing the observed univariate
                          Local Geary values.
        p_sim           : numpy array
                          array containing the simulated
                          p-values for each unit.
        labs            : numpy array
                          array containing the labels for if each observation.
        """

        self.connectivity = connectivity
        self.labels = labels
        self.sig = sig
        self.permutations = permutations
        self.n_jobs = n_jobs
        self.keep_simulations = keep_simulations
        self.seed = seed
        self.island_weight = island_weight
        self.drop_islands = drop_islands





    def fit(self, x):
        """
        Parameters
        ----------
        x                : numpy.ndarray
                           array containing continuous data

        Returns
        -------
        the fitted estimator.

        Notes
        -----
        Technical details and derivations can be found in :cite:`Anselin1995`.

        Examples
        --------
        Guerry data replication GeoDa tutorial
        >>> import libpysal as lp
        >>> import geopandas as gpd
        >>> guerry = lp.examples.load_example('Guerry')
        >>> guerry_ds = gpd.read_file(guerry.get_path('Guerry.shp'))
        >>> w = libpysal.weights.Queen.from_dataframe(guerry_ds)
        >>> y = guerry_ds['Donatns']
        >>> lG = Local_Geary(connectivity=w).fit(y)
        >>> lG.localG[0:5]
        >>> lG.p_sim[0:5]
        """
        x = np.asarray(x).flatten()

        w = self.connectivity
        w.transform = "r"

        permutations = self.permutations
        sig = self.sig
        keep_simulations = self.keep_simulations
        n_jobs = self.n_jobs

        self.localG = self._statistic(x, w, self.drop_islands)

        if permutations:
            self.p_sim, self.rlocalG = _crand_plus(
                z=(x - np.mean(x)) / np.std(x),
                w=w,
                observed=self.localG,
                permutations=permutations,
                keep=keep_simulations,
                n_jobs=n_jobs,
                stat_func=_local_geary,
                island_weight=self.island_weight,
            )

        if self.labels:
            Eij_mean = np.mean(self.localG)
            x_mean = np.mean(x)
            # Create empty vector to fill
            self.labs = np.empty(len(x)) * np.nan
            # Outliers
            locg_lt_eij = self.localG < Eij_mean
            p_leq_sig = self.p_sim <= sig
            self.labs[locg_lt_eij & (x > x_mean) & p_leq_sig] = 1
            # Clusters
            self.labs[locg_lt_eij & (x < x_mean) & p_leq_sig] = 2
            # Other
            self.labs[(self.localG > Eij_mean) & p_leq_sig] = 3
            # Non-significant
            self.labs[self.p_sim > sig] = 4

        return self



    @staticmethod
    def _statistic(x, w, drop_islands):

        # Caclulate z-scores for x
        zscore_x = (x - np.mean(x)) / np.std(x)
        # Create focal (xi) and neighbor (zi) values
        adj_list = w.to_adjlist(remove_symmetric=False, drop_islands=drop_islands)
        zseries = pd.Series(zscore_x, index=w.id_order)
        zi = zseries.loc[adj_list.focal].values
        zj = zseries.loc[adj_list.neighbor].values
        # Carry out local Geary calculation
        gs = adj_list.weight.values * (zi - zj) ** 2
        # Reorganize data
        adj_list_gs = pd.DataFrame(adj_list.focal.values, gs).reset_index()
        adj_list_gs.columns = ["gs", "ID"]
        adj_list_gs = adj_list_gs.groupby(by="ID").sum()
        # Rearrange data based on w id order            
        
        adj_list_gs["w_order"] = w.id_order
        adj_list_gs.sort_values(by="w_order", inplace=True)

        localG = adj_list_gs.gs.values

        return localG




# --------------------------------------------------------------
# Conditional Randomization Function Implementations
# --------------------------------------------------------------

# Note: does not using the scaling parameter


@_njit(fastmath=True)
def _local_geary(i, z, permuted_ids, weights_i, scaling):
    other_weights = weights_i[1:]
    zi, zrand = _prepare_univariate(i, z, permuted_ids, other_weights)
    return (zi - zrand) ** 2 @ other_weights

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