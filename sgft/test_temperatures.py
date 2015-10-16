from __future__ import absolute_import
import graph_tool.draw as gt_draw
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn.apionly as sns
import numpy as np
import pickle
import os
from scipy.spatial.distance import pdist, squareform
import sgft.temperatures as temperatures
import sgft.graph_spectrogram as spec
import sgft.utils as utils


def show_spectrogram(s, cmap):
    s /= np.atleast_2d(np.max(s, axis=0))
    s_range = np.max(s) - np.min(s)
    s = utils.smoothstep(s,
                         min_edge=np.min(s) + s_range / 3,
                         max_edge=np.max(s) - s_range / 10)
    spec.plot(s, cmap)


def process_us():
    name = 'US'
    year = 2014
    n_neigh = 6

    graph = temperatures.country_network(name, 2014, n_neigh=6)
    pos = graph.vertex_properties['pos']
    station_values = graph.vertex_properties['station_values']
    weight = graph.edge_properties['weights']
    x_signal = station_values.a

    n_eigs = 500
    # n_eigs = graph.num_vertices() - 1
    alpha = -1e-3

    file_name = '{0}_{1}_k{2}'
    file_name = file_name.format(name, year, n_neigh)
    if os.path.exists(file_name + '_spec.pickle'):
        with open(file_name + '_spec.pickle', 'r') as f:
            spec_weighted = pickle.load(f)
            factory = pickle.load(f)
    else:
        factory = spec.PageRankSGFT(graph, n_eigs, alpha, weight=weight)
        spec_weighted = factory.compute(x_signal)
        with open(file_name + '_spec.pickle', 'w') as f:
            pickle.dump(spec_weighted, f)
            pickle.dump(factory, f)

    palette = sns.cubehelix_palette(256, start=2, rot=0, dark=0.15, light=1)
    cmap = colors.ListedColormap(palette)
    plt.figure()
    show_spectrogram(spec_weighted[0:30, :], cmap=cmap)
    plt.savefig(file_name + '_spec.pdf', dpi=300)

    temperatures.plot(graph, weight, pos, station_values, file_name)
    spec.show_window(factory, .5 * (graph.num_vertices() + 1), weight, pos,
                     file_name + '_window1.png')
    spec.show_window(factory, .25 * (graph.num_vertices() + 1), weight, pos,
                     file_name + '_window2.png')
    spec.show_window(factory, .75 * (graph.num_vertices() + 1), weight, pos,
                     file_name + '_window3.png')
    show_spec_in_graph(graph, 417, spec_weighted, pos, weight,
                       file_name + '_florida')


def show_spec_in_graph(graph, vertex, spec, pos, weight, file_name):
    dist = 1.0 - squareform(pdist(spec.T, 'cosine'))

    plt.figure()
    plt.stem(dist[vertex, :], markerfmt=' ')

    rim = graph.new_vertex_property('vector<double>')
    rim.set_2d_array(np.array([0, 0, 0, 1]))
    rim[graph.vertex(vertex)] = [0.8941176471, 0.1019607843, 0.1098039216, 1]
    rim_width = graph.new_vertex_property('float', vals=0.5)
    rim_width.a[vertex] = 2
    shape = graph.new_vertex_property('int', vals=0)
    shape[graph.vertex(vertex)] = 2
    size = graph.new_vertex_property('double', vals=10)
    size.a[vertex] = 15
    correlation = graph.new_vertex_property('double', vals=2)
    correlation.a = dist[vertex, :]
    vorder = graph.new_vertex_property('int', vals=0)
    vorder.a[vertex] = 1

    palette = sns.cubehelix_palette(256)
    cmap = colors.ListedColormap(palette)
    gt_draw.graph_draw(graph, pos=pos, vertex_color=rim, vorder=vorder,
                       vertex_pen_width=rim_width,
                       vertex_shape=shape, vertex_fill_color=correlation,
                       vcmap=cmap, vertex_size=size, edge_color=[0, 0, 0, 0.7],
                       edge_pen_width=weight, output=file_name + '.png',
                       output_size=(1200, 1200))

    plt.figure()
    utils.plot_colorbar(cmap, np.arange(0, 1.01, 0.2), file_name)


if __name__ == '__main__':
    process_us()
    plt.show()
