from __future__ import absolute_import
import graph_tool.generation as gt_gen
import graph_tool.draw as gt_draw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn.apionly as sns
import sgft.graph_spectrogram as spec
import sgft.comparison


def test_chain():
    periodic = True
    # periodic = False
    graph = gt_gen.lattice([1, 200], periodic=periodic)
    if periodic:
        figure_title = 'line_periodic'
    else:
        figure_title = 'line_non-periodic'

    n = graph.num_vertices()
    location1 = 0.2 * n
    location2 = n - location1
    jump = 1e-3
    weight = graph.new_edge_property('double', vals=1)
    e1 = graph.edge(location1, location1 + 1)
    weight[e1] = jump
    e2 = graph.edge(location2 - 1, location2)
    weight[e2] = jump

    pos_x = np.arange(n)
    pos_y = np.zeros((n,))
    v_pos = graph.new_vertex_property('vector<double>',
                                      vals=np.vstack((pos_x, pos_y)).T)
    v_text = graph.new_vertex_property('string')
    for v in graph.vertices():
        v_text[v] = pos_x[graph.vertex_index[v]]
    palette = sns.color_palette('Set1', n_colors=2)
    cmap = colors.ListedColormap(palette)
    gt_draw.graph_draw(graph, pos=v_pos, edge_color=weight, ecmap=cmap,
                       edge_pen_width=.5, vertex_fill_color='w', vertex_size=2,
                       vertex_text=v_text, vertex_font_size=1,
                       output=figure_title + '_graph.pdf')

    x_signal1 = np.cos(np.linspace(0, 4 * np.pi, location1))
    x_signal2 = np.cos(np.linspace(0, 50 * np.pi, n - 2 * location1))
    x_signal3 = np.cos(np.linspace(0, 8 * np.pi, location1))
    x_signal = np.hstack([x_signal1, x_signal2, x_signal3])

    palette = sns.color_palette('Set1', n_colors=3)
    plt.figure()
    markerline, stemlines, baseline = plt.stem(x_signal, markerfmt=' ')
    plt.setp(stemlines, color=palette[1], linewidth=1.5)
    plt.setp(baseline, color='k')
    plt.savefig(figure_title + '_signal.pdf', dpi=300)

    n_eigs = graph.num_vertices() - 1
    tau = 200
    alpha = -1e-4

    factories = [spec.ConvolutionSGFT(graph, n_eigs, tau, weight=weight),
                 spec.PageRankSGFT(graph, n_eigs, alpha, weight=weight),
                 spec.ConvolutionSGFT(graph, n_eigs, tau, weight=None),
                 spec.PageRankSGFT(graph, n_eigs, alpha, weight=None)]

    sgft.comparison.compare_spectrograms(factories, x_signal, graph, v_pos,
                                         file_name=figure_title)
    sgft.comparison.compare_localization(factories, location1, graph, v_pos,
                                         file_name=figure_title)


if __name__ == '__main__':
    test_chain()
    plt.show()
