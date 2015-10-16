import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform
import graph_tool as gt
import graph_tool.draw as gt_draw
import gzip
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn.apionly as sns
import sgft.utils as utils

folder_name = os.path.dirname(__file__)


def country_network(name, year, check_fields={}, n_neigh=6):
    graph_file_name = name + '_' + str(year) + '_k' + str(n_neigh) + '.xml'
    if os.path.exists(graph_file_name):
        return gt.load_graph(graph_file_name)

    if name == 'US' and check_fields == {}:
        states = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                  'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                  'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ',
                  'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD',
                  'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
        check_fields = {'STATE': states}

    station_codes = []
    pos = []
    with open(os.path.join(folder_name, 'isd-history.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            functional = int(row['BEGIN'][:4]) <= year <= int(row['END'][:4])
            lon = row['LON']
            lat = row['LAT']
            if row['CTRY'] == name and functional and lon and lat:
                if not all([row[field] in check_fields[field]
                            for field in check_fields]):
                    continue
                station_codes.append((row['USAF'], row['WBAN']))
                pos.append([float(lon), -float(lat)])

    station_values, missing = read_stations(station_codes, year)
    pos = np.delete(np.array(pos), missing, axis=0)

    weights = squareform(pdist(pos))
    weights = np.exp(-weights / np.median(weights))
    idx_sorted = np.argsort(weights)
    idx_sorted = idx_sorted[:, -n_neigh-1:-1]
    print idx_sorted.shape

    graph = gt.Graph(directed=False)
    graph.add_vertex(n=pos.shape[0])
    e_weights = graph.new_edge_property('double', vals=1)
    for i in range(weights.shape[0]):
        w_i = weights[i, idx_sorted[i, :]]
        for j in range(n_neigh):
            e = graph.edge(i, idx_sorted[i, j], new=True)
            e_weights[e] = w_i[j]

    v_pos = graph.new_vertex_property('vector<double>', vals=pos)
    v_values = graph.new_vertex_property('double', vals=station_values)

    graph.vertex_properties['pos'] = v_pos
    graph.vertex_properties['station_values'] = v_values
    graph.edge_properties['weights'] = e_weights

    graph.save(graph_file_name)

    return graph


def read_stations(codes, year):
    dir_name = os.path.join(folder_name, 'gsod_' + str(year))
    missing = []
    station_values = []
    for i, c in enumerate(codes):
        file_name = c[0] + '-' + c[1] + '-' + str(year) + '.op.gz'
        file_name = os.path.join(dir_name, file_name)
        if not os.path.exists(file_name):
            missing.append(i)
            continue
        with gzip.open(file_name, 'r') as f:
            values = []
            counts = []
            for k, line in enumerate(f):
                if k == 0:
                    continue
                tokens = line.split()
                if tokens[3] == '9999.9':
                    continue
                values.append(float(tokens[3]))
                counts.append(int(tokens[4]))
            values = np.array(values)
            counts = np.array(counts)
            mean_val = np.sum(values * counts) / np.sum(counts)
            station_values.append(mean_val)

    return np.array(station_values), missing


def plot(graph, weights, pos, station_values, name):
    palette = sns.color_palette('RdBu', n_colors=256)
    cmap = colors.ListedColormap(palette[::-1])

    weights = weights.copy()
    weights.a -= np.min(weights.a)
    weights.a *= 2 / np.max(weights.a)
    weights.a += 0.2

    gt_draw.graph_draw(graph, pos=pos, vertex_color=[0, 0, 0, 0.5],
                       vertex_fill_color=station_values, vcmap=cmap,
                       vertex_size=5, edge_color=[0, 0, 0, 0.7],
                       edge_pen_width=weights,
                       output=name + '_temp.svg')
    gt_draw.graph_draw(graph, pos=pos, vertex_color=[0, 0, 0, 0.5],
                       vertex_fill_color=station_values, vcmap=cmap,
                       vertex_size=10, edge_color=[0, 0, 0, 0.7],
                       edge_pen_width=weights,
                       output=name + '_temp.png', output_size=(1200, 1200))

    min_val = np.min(station_values.a)
    max_val = np.max(station_values.a)
    step = (max_val - min_val) / 5
    labels = np.array(['{0:.2f}'.format(x)
                       for x in np.arange(min_val, max_val, step)])
    plt.figure()
    utils.plot_colorbar(cmap, labels, name)
