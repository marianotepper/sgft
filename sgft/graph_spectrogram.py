from abc import ABCMeta, abstractmethod
import numpy as np
import timeit
from graph_tool import spectral as gt_spectral
import graph_tool.draw as gt_draw
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn.apionly as sns
import sgft.ppr as ppr
import sgft.utils as utils


class SGFT:
    __metaclass__ = ABCMeta

    def __init__(self, graph, n_eigs):
        self.graph = graph
        self.basis = None
        self.n_eigs = n_eigs
        self.n = graph.num_vertices()
        self.normalization_constant = 0

    @abstractmethod
    def get_window(self, vertex):
        pass

    def compute(self, x_signal):
        spec = np.zeros((self.n_eigs, self.n))

        t = timeit.default_timer()
        for v in range(self.n):
            window = self.get_window(v)
            window = np.atleast_2d(window).T
            modulated = self.normalization_constant * np.multiply(window,
                                                                  self.basis)
            spec[:, v] = x_signal.T.dot(modulated)

        print timeit.default_timer() - t
        return np.power(spec, 2)


class ConvolutionSGFT(SGFT):

    def __init__(self, graph, n_eigs, tau, weight=None):
        SGFT.__init__(self, graph, n_eigs)
        lap = gt_spectral.laplacian(graph, normalized=False, weight=weight)
        eig_val, self.basis = eigsh(lap, n_eigs, which='SM')
        self.normalization_constant = np.power(graph.num_vertices(), 0.5)
        self.g_hat = np.exp(-tau * eig_val)

    def get_window(self, vertex):
        window = np.dot(self.basis, np.multiply(self.basis[vertex, :],
                                                self.g_hat))
        return window / np.sum(window)

    def compute(self, x_signal):
        spec = np.zeros((self.n_eigs, self.n))

        t = timeit.default_timer()
        for v in range(self.n):
            window = self.get_window(v)
            window = np.atleast_2d(window).T
            modulated = self.normalization_constant * np.multiply(window,
                                                                  self.basis)
            spec[:, v] = x_signal.T.dot(modulated)

        print timeit.default_timer() - t
        return np.power(spec, 2)


class PageRankSGFT(SGFT):
    def __init__(self, graph, n_eigs, alpha, weight=None):
        SGFT.__init__(self, graph, n_eigs)
        self.graph = graph
        self.factory = ppr.PersonalizedPageRank(graph, n_eigs, weight=weight)
        self.alpha = alpha + self.factory.eig_val[0]
        self.basis = self.factory.basis
        vol = utils.graph_volume(graph, weight)
        self.normalization_constant = np.power(vol, 0.5)

    def get_window(self, vertex):
        return self.factory.vector([self.graph.vertex(vertex)], self.alpha)[0]


def plot(spec, cmap):
    plt.imshow(spec, interpolation='none', cmap=cmap, origin='lower')
    plt.colorbar()
    plt.axis('tight')


def plot_sparsity(spec, threshold):
    plt.stem(np.sum(spec > np.max(spec) * threshold, axis=0))


def show_window(spectrogram, v, weight=None, pos=None, file_name=None,
                vertex_size=10):
    window = spectrogram.get_window(v)
    window = spectrogram.graph.new_vertex_property('double', vals=window)

    if weight is None:
        edge_pen_width = 1.0
    else:
        edge_pen_width = weight

    palette = sns.color_palette('YlOrRd', n_colors=256)
    cmap = colors.ListedColormap(palette)
    gt_draw.graph_draw(spectrogram.graph, pos=pos, vertex_color=[0, 0, 0, 0.5],
                       vertex_fill_color=window, vcmap=cmap,
                       vertex_size=vertex_size, edge_color=[0, 0, 0, 0.7],
                       edge_pen_width=edge_pen_width, output=file_name,
                       output_size=(1200, 1200))