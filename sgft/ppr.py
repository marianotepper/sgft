from __future__ import absolute_import
from graph_tool import spectral as gt_spectral
import graph_tool.draw as gt_draw
import numpy as np
from scipy.sparse.linalg import eigsh
import math
import matplotlib.colors as colors
import seaborn.apionly as sns
import sgft.utils as utils


class PersonalizedPageRank:

    def __init__(self, graph, k, weight=None):
        if k >= graph.num_vertices():
            raise ValueError('k must be in the range ' +
                             '[0, graph.num_vertices()). ' +
                             'This is due to scipy.sparse.linalg restrictions')

        self.lap = gt_spectral.laplacian(graph, normalized=True, weight=weight)
        self.adj = gt_spectral.adjacency(graph, weight=weight)
        self._deg_vec = np.asarray(self.adj.sum(axis=1))

        self._weight = weight

        self._vol_graph = utils.graph_volume(graph, self._weight)
        self._n = graph.num_vertices()
        self._vertex_index = graph.vertex_index

        self.eig_val, self.eig_vec = eigsh(self.lap, k, which='SM')

        self._deg_vec = np.asarray(self.adj.sum(axis=1))
        self.basis = np.multiply(np.power(self._deg_vec, -0.5), self.eig_vec)
        self._deg_vec = np.squeeze(self._deg_vec)

    def vector(self, seeds, gamma):

        vol_seeds = utils.volume(seeds, self._weight)
        vol_complement_seeds = self._vol_graph - vol_seeds

        temp = math.sqrt(vol_seeds * vol_complement_seeds / self._vol_graph)
        seed_vector = np.ones((self._n,))
        seed_vector *= -temp / vol_complement_seeds
        for s in seeds:
            idx = self._vertex_index[s]
            seed_vector[idx] = temp / vol_seeds

        ds = np.multiply(self._deg_vec, seed_vector)
        weights = self.basis.T.dot(ds) / (self.eig_val - gamma)
        # By definition,
        # (1) s.T * D * 1 = 0
        # (2) self.dv[:, 0] is a constant vector
        # Hence:
        weights[0] = 0

        x = self.basis.dot(weights)
        x[x < 0] = 0
        x /= np.sum(x)
        return x, weights
