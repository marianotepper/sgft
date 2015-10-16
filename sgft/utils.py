from __future__ import absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt

_sew_name = 'sum_edge_weights'


def graph_volume(graph, weight=None):
    name = _sew_name
    if weight is None:
        name += '_unweighted'
    else:
        name += '_weighted'
    if name not in graph.graph_properties:
        if weight is None:
            sew = 2 * graph.num_edges()
        else:
            sew = 2 * sum([weight[e] for e in graph.edges()])
        graph.graph_properties[name] = graph.new_graph_property('float',
                                                                val=sew)
    return graph.graph_properties[name]


def volume(bunch, weight):
    """
    Returns the sum of the costs (weights) of the outgoing edges from a
    set of nodes.

    :param bunch: A container of nodes.
    :type bunch: iterable container
    :return: an nonnegative integer for unweighted graphs. For weighted
    graphs, its type depends on the type of the weighting key.
    """
    sew = 0
    for v in bunch:
        sew += v.out_degree(weight=weight)
    return sew


def plot_colorbar(cmap, labels, file_name):
    gradient = np.linspace(0, 1, 256)
    gradient = np.repeat(np.atleast_2d(gradient), 10, axis=0)

    plt.imshow(gradient, cmap=cmap)
    plt.xticks(np.linspace(-0.5, gradient.shape[1]-0.5, labels.shape[0]),
               labels)
    plt.tick_params(
        axis='y',
        which='both',
        left='off',
        right='off',
        labelleft='off',
        labelright='off')
    plt.savefig(file_name + '_colorbar.pdf')


def smoothstep(x, min_edge=0, max_edge=1):
    x = np.clip((x - min_edge)/(max_edge - min_edge), 0.0, 1.0)
    return x * x * (3 - 2*x)
