from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn.apionly as sns
import graph_tool.draw as gt_draw
import sgft.graph_spectrogram as spec
import sgft.utils as utils


def compare_spectrograms(factories, x_signal, graph, pos, weight=None,
                         file_name=None, show_ncomps=None):
    palette = sns.cubehelix_palette(256, start=2, rot=0, dark=0.15, light=1)
    cmap = colors.ListedColormap(palette)
    stem_palette = sns.color_palette('Set1', n_colors=2)

    def show_spectrogram(s):
        s /= np.atleast_2d(np.max(s, axis=0))
        s_range = np.max(s) - np.min(s)
        s = utils.smoothstep(s,
                             min_edge=np.min(s) + s_range / 3,
                             max_edge=np.max(s) - s_range / 10)
        if show_ncomps is None:
            spec.plot(s, cmap)
        else:
            spec.plot(s[:show_ncomps, :], cmap)

    def show_argmax_spectrogram_graph(s, vertex_size=20, amax_file_name=None):
        amax = np.argmax(np.abs(s), axis=0)
        n_values = np.unique(amax).size
        assignment = graph.new_vertex_property('double', vals=amax)

        if weight is None:
            edge_pen_width = 1.0
        else:
            edge_pen_width = weight

        palette = sns.color_palette('BuGn', n_colors=n_values)
        cmap = colors.ListedColormap(palette)
        gt_draw.graph_draw(graph, pos=pos, vertex_color=[0, 0, 0, 0.5],
                           vertex_fill_color=assignment, vcmap=cmap,
                           vertex_size=vertex_size, edge_color=[0, 0, 0, 0.7],
                           edge_pen_width=edge_pen_width, output=amax_file_name,
                           output_size=(1200, 1200))

    def show_argmax_spectrogram_1d(s):
        amax = np.argmax(np.abs(s), axis=0)
        _, stemlines, baseline = plt.stem(amax, markerfmt=' ')
        plt.setp(stemlines, 'color', stem_palette[1])
        plt.setp(baseline, 'color','k')
        plt.ylim((0, s.shape[0]))

    if file_name is None:
        plt.figure()
        for i in range(len(factories)):
            spectrogram = factories[i].compute(x_signal)
            plt.subplot(2, 4, i + 1)
            show_spectrogram(spectrogram)
            plt.subplot(2, 4, 5 + i)
            show_argmax_spectrogram_1d(spectrogram)
    else:
        file_name += '_{0}_{1}'
        for i in range(len(factories)):
            spectrogram = factories[i].compute(x_signal)
            plt.figure()
            show_spectrogram(spectrogram)
            plt.savefig(file_name.format(i, 'spec.pdf'), dpi=300)
            plt.figure()
            show_argmax_spectrogram_1d(spectrogram)
            plt.savefig(file_name.format(i, 'spec_amax.pdf'), dpi=300)
            amax_file_name = file_name.format(i, 'spec_amax.png')
            show_argmax_spectrogram_graph(spectrogram,
                                          amax_file_name=amax_file_name)


def compare_localization(factories, loc, file_name=None):
    locations = [k + loc for k in range(5, 20, 5)]
    palette = sns.color_palette('Set1', n_colors=len(locations))

    if file_name is not None:
        file_name += '_spec{0}.pdf'
    for i in range(len(factories)):
        plt.figure()
        plt.hold(True)
        max_val = -1
        for j, v in enumerate(locations):
            window = factories[i].get_window(v)
            max_val = max(max_val, np.max(window))
            plt.plot(window, color=palette[j], linewidth=3)
            plt.plot([v, v], [0, window[v] * 1], color=palette[j],
                     linestyle='--', linewidth=3)
        lim = (np.floor(max_val / 0.005) + 1) * 0.005
        if lim - max_val < 0.002:
            lim += 0.005
        plt.ylim(0, lim)
        if file_name is not None:
            plt.savefig(file_name.format(i), dpi=300)
