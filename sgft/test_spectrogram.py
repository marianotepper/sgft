from __future__ import absolute_import
import graph_tool.generation as gt_gen
# import graph_tool.draw as gt_draw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn.apionly as sns
import timeit
import sgft.ppr as ppr


def test_chain():
    graph = gt_gen.lattice([1, 1000], periodic=False)
    # gt_draw.graph_draw(graph)

    jump = 1e-3
    weight = graph.new_edge_property('double', vals=1)
    e1 = graph.edge(300, 301)
    weight[e1] = jump
    e1 = graph.edge(699, 700)
    weight[e1] = jump# + 0.01

    for e in graph.edges():
        # weight[e] += min(abs(0.3 * np.random.randn()), 1)
        if e.source() == 300 or e.source() == 699:
            print e, weight[e]

    n_eigs = 100
    alpha = -1e-10

    factory_weighted = ppr.PersonalizedPageRank(graph, n_eigs, weight=weight)
    factory_unweighted = ppr.PersonalizedPageRank(graph, n_eigs, weight=None)

    spectrogram_weighted = np.zeros((graph.num_vertices(), n_eigs))
    spectrogram_unweighted = np.zeros((graph.num_vertices(), n_eigs))
    t = timeit.default_timer()
    for v in range(graph.num_vertices()):
        _, loadings_weighted = factory_weighted.vector([graph.vertex(v)], alpha)
        _, loadings_unweighted = factory_unweighted.vector([graph.vertex(v)],
                                                           alpha)
        spectrogram_weighted[v, :] = loadings_weighted
        spectrogram_unweighted[v, :] = loadings_unweighted
    print timeit.default_timer() - t
    spectrogram_weighted = np.abs(spectrogram_weighted)
    spectrogram_unweighted = np.abs(spectrogram_unweighted)
    spectrogram_weighted /= np.atleast_2d(np.sum(spectrogram_weighted, axis=1)).T
    spectrogram_unweighted /= np.atleast_2d(np.sum(spectrogram_unweighted, axis=1)).T

    palette = sns.color_palette('RdBu_r', n_colors=256)
    cmap = colors.ListedColormap(palette, N=256)

    plt.figure()
    plt.subplot(121)
    plt.imshow(spectrogram_weighted, interpolation='none', cmap=cmap)
    plt.title('Weighted')
    plt.axis('tight')
    plt.subplot(122)
    plt.imshow(spectrogram_unweighted, interpolation='none', cmap=cmap)
    plt.title('Unweighted')
    plt.axis('tight')


def test_circle():
    size = 100
    graph = gt_gen.lattice([size, size], periodic=False)


    indices = np.arange(size * size)
    rows = np.mod(indices, size)
    cols = np.floor(indices / size)
    v_pos = graph.new_vertex_property('vector<double>',
                                      vals=np.vstack((rows, cols)).T)

    center = size * (size / 2) + size / 2 - 20
    radius = 20

    vertex_w = set([])
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            if i ** 2 + j ** 2 < radius ** 2:
                idx = j * size + i + center
                vertex_w.add(idx)
                idx = j * size + i + center + 35
                vertex_w.add(idx)
    print len(vertex_w)

    jump = 1e-5
    weight = graph.new_edge_property('double', vals=1)
    for idx in vertex_w:
        u = graph.vertex(idx)
        for e in u.out_edges():
            if e.target() not in vertex_w:
                weight[e] = jump
    # for e in graph.edges():
    #     weight[e] += min(abs(0.01 * np.random.randn()), 1)

    v_color = graph.new_vertex_property('vector<double>')
    for u in graph.vertices():
        if graph.vertex_index[u] in vertex_w:
            v_color[u] = [1, 0, 0, 1]
        else:
            v_color[u] = [0, 0, 1, 1]
    # gt_draw.graph_draw(graph, pos=v_pos, vertex_fill_color=v_color,
    #                    vertex_size=3)

    alpha = -1e-5
    n_eigs = 50

    factory_weighted = ppr.PersonalizedPageRank(graph, n_eigs, weight=weight)
    factory_unweighted = ppr.PersonalizedPageRank(graph, n_eigs, weight=None)

    spectrogram_weighted = np.zeros((graph.num_vertices(), n_eigs - 1))
    spectrogram_unweighted = np.zeros((graph.num_vertices(), n_eigs - 1))
    t = timeit.default_timer()
    for v in range(graph.num_vertices()):
        _, loadings_weighted = factory_weighted.vector([graph.vertex(v)], alpha)
        _, loadings_unweighted = factory_unweighted.vector([graph.vertex(v)],
                                                           alpha)
        spectrogram_weighted[v, :] = loadings_weighted
        spectrogram_unweighted[v, :] = loadings_unweighted
    print timeit.default_timer() - t
    spectrogram_weighted = np.abs(spectrogram_weighted)
    spectrogram_unweighted = np.abs(spectrogram_unweighted)
    spectrogram_weighted /= np.atleast_2d(np.sum(spectrogram_weighted, axis=1)).T
    spectrogram_unweighted /= np.atleast_2d(np.sum(spectrogram_unweighted, axis=1)).T

    plt.figure()
    plt.subplot(121)
    plt.imshow(spectrogram_weighted, interpolation='none')
    plt.title('Weighted')
    plt.axis('tight')
    plt.subplot(122)
    plt.imshow(spectrogram_unweighted, interpolation='none')
    plt.title('Unweighted')
    plt.axis('tight')


if __name__ == '__main__':
    plt.switch_backend('TkAgg')  # otherwise, monospace fonts do not work in mac
    test_chain()
    # test_circle()
    plt.show()