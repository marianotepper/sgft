from __future__ import absolute_import
import graph_tool.generation as gt_gen
import graph_tool.draw as gt_draw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn.apionly as sns
import sgft.graph_spectrogram as spec
import sgft.comparison


def test_circle():
    size = 50
    periodic = True
    # periodic = False
    graph = gt_gen.lattice([size, size], periodic=periodic)
    if periodic:
        figure_title = 'grid_periodic'
    else:
        figure_title = 'grid_non-periodic'

    indices = np.arange(size * size)
    rows = np.mod(indices, size)
    cols = np.floor(indices / size)
    v_pos = graph.new_vertex_property('vector<double>',
                                      vals=np.vstack((rows, cols)).T)

    radius = 10
    center = (size + 1) * (size / 2)

    vertex_w = set([])
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            if i ** 2 + j ** 2 < radius ** 2:
                idx = j * size + i + center
                vertex_w.add(idx)

    jump = 1e-5
    weight = graph.new_edge_property('double', vals=1)
    for idx in vertex_w:
        u = graph.vertex(idx)
        for e in u.out_edges():
            if e.target() not in vertex_w:
                weight[e] = jump

    v_color = graph.new_vertex_property('vector<double>')
    for u in graph.vertices():
        if graph.vertex_index[u] in vertex_w:
            v_color[u] = [1, 0, 0, 1]
        else:
            v_color[u] = [0, 0, 1, 1]

    vertex_w = list(vertex_w)

    x = np.linspace(-np.pi, np.pi, size)
    y = np.linspace(-np.pi, np.pi, size)
    xx, yy = np.meshgrid(x, y)
    z1 = np.sin(np.pi * xx).flatten()
    z2 = np.sin(2 * np.pi * yy).flatten()
    x_signal = z1
    x_signal[vertex_w] = z2[vertex_w]

    palette = sns.color_palette('RdBu', n_colors=256, desat=.7)
    cmap = colors.ListedColormap(palette, N=256)
    plt.figure()
    plt.imshow(np.reshape(x_signal, (size, size)), interpolation='nearest',
               cmap=cmap)
    plt.savefig(figure_title + '_signal.pdf', dpi=300)

    n_eigs = 500  # graph.num_vertices() - 1
    alpha = -1e-4

    factories = [spec.ConvolutionSGFT(graph, n_eigs, tau=200, weight=weight),
                 spec.PageRankSGFT(graph, n_eigs, alpha, weight=weight),
                 spec.ConvolutionSGFT(graph, n_eigs, tau=5, weight=None),
                 spec.PageRankSGFT(graph, n_eigs, alpha, weight=None)]

    sgft.comparison.compare_spectrograms(factories, x_signal,
                                         graph, v_pos,
                                         file_name=figure_title,
                                         show_ncomps=300)
    spec.show_window(factories[0], center, weight=weight, pos=v_pos,
                     vertex_size=20,
                     file_name=figure_title + '_w_window.png')
    spec.show_window(factories[1], center, weight=weight, pos=v_pos,
                     vertex_size=20,
                     file_name=figure_title + '_w_window_ppr.png')
    spec.show_window(factories[2], center, weight=None, pos=v_pos,
                     vertex_size=20,
                     file_name=figure_title + '_u_window.png')
    spec.show_window(factories[3], center, weight=None, pos=v_pos,
                     vertex_size=20,
                     file_name=figure_title + '_u_window_ppr.png')

if __name__ == '__main__':
    test_circle()
    plt.show()