

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from networkx.drawing.nx_pydot import graphviz_layout

def convert_to_nx(data):
    # convert PyG graph to a networkx graph manually
    # since the to_networkx method is bugged
    G = nx.Graph()
    G.add_nodes_from(range(len(data.x)))
    G.add_edges_from(data.edge_index.T.numpy())
    for i in range(len(data.x)):
        G.nodes[i]['x'] = data.x[i].numpy()
    return G

def create_nx_graph(halo_id, halo_desc_id, halo_props=None):
    """ Create a directed graph of the halo merger tree.

    Parameters
    ----------
    halo_id : array_like
        Array of halo IDs.
    halo_desc_id : array_like
        Array of halo descendant IDs.
    halo_props : dict or None, optional
        Array of halo properties. If provided, the properties will be added as
        node attributes.

    Returns
    -------
    G : networkx.DiGraph
        A directed graph of the halo merger tree.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes using indices
    for idx in range(len(halo_id)):
        if halo_props is not None:
            prop_dict = {key: halo_props[key][idx] for key in halo_props.keys()}
            G.add_node(idx, **prop_dict)
        G.add_node(idx)

    # Add edges based on desc_id
    for idx, desc_id in enumerate(halo_desc_id):
        if desc_id != -1:
            parent_idx = np.where(halo_id==desc_id)[0][0] # Find the index of the parent ID
            G.add_edge(parent_idx, idx) # Use indices for edges
    return G

def plot_graph(G, fig_args=None, draw_args=None):
    if isinstance(G, Data):
        G = convert_to_nx(G)

    pos = graphviz_layout(G, prog='dot')
    fig_args = fig_args or {}
    draw_args = draw_args or {}

    fig, ax = plt.subplots(**fig_args)

    default_draw_args = dict(
        with_labels=False,
        node_size=20, node_color="black",
        font_size=10, font_color="black"
    )
    default_draw_args.update(draw_args)
    nx.draw(G, pos, ax=ax, **default_draw_args)
    return fig, ax

def dfs(edge_index, start_node=0):
    """ Perform a depth-first search on the graph and return the order of nodes """
    # Convert edge_index to a graph representation
    graph = {}
    for i in range(edge_index.size(1)):
        src, dest = edge_index[:, i].tolist()
        if src in graph:
            graph[src].add(dest)
        else:
            graph[src] = {dest}
        if dest in graph:
            graph[dest].add(src)
        else:
            graph[dest] = {src}

    # DFS algorithm
    def dfs_visit(node, visited):
        visited.add(node)
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs_visit(neighbor, visited)

    visited = set()
    order = []
    dfs_visit(start_node, visited)
    return order

def get_main_branch(mass, edge_index):
    # follow the main branch and get index
    main_branch_index = [0]
    curr_index = 0
    while True:
        # get all progenitors
        prog_index = edge_index[1][edge_index[0] == curr_index]

        # if no progenitors, break
        if len(prog_index) == 0:
            break

        # get the progenitor with the highest mass
        prog_mass = mass[prog_index]
        prog_index = prog_index[prog_mass.argmax()].item()

        main_branch_index.append(prog_index)
        curr_index = prog_index

    return main_branch_index


def subsample_trees(
    halo_ids, halo_desc_ids, node_feats, snap_nums, new_snap_nums):
    """ Subsample the trees to only include the snapshots in new_snap_nums.
    """
    new_node_feats = []
    new_halo_ids = []
    new_halo_desc_ids = []
    for i in range(len(snap_nums)):
        if snap_nums[i] in new_snap_nums:
            new_node_feats.append(node_feats[i])
            new_halo_ids.append(halo_ids[i])
            new_halo_desc_ids.append(halo_desc_ids[i])
    if len(new_node_feats) == 0:
        return new_node_feats, new_halo_ids, new_halo_desc_ids
    new_node_feats = np.stack(new_node_feats, axis=0)
    new_halo_ids = np.array(new_halo_ids)
    new_halo_desc_ids = np.array(new_halo_desc_ids)

    # update the halo_desc_ids
    # iterate over all halos
    id_to_index = {halo_id: i for i, halo_id in enumerate(halo_ids)}
    for i in range(len(new_halo_desc_ids)):
        halo_desc_id = new_halo_desc_ids[i]
        if halo_desc_id == -1:
            continue
        while halo_desc_id not in new_halo_ids:
            halo_desc_id = halo_desc_ids[id_to_index[halo_desc_id]]
        new_halo_desc_ids[i] = halo_desc_id

    return new_halo_ids, new_halo_desc_ids, new_node_feats

def calc_num_progenitors(halo_ids, halo_desc_ids):
    """ Calculate the number of progenitors for each halo. """
    unique_desc_ids, counts = np.unique(halo_desc_ids, return_counts=True)
    num_progenitors = np.zeros(len(halo_desc_ids), dtype=np.int32)
    for desc_id, count in zip(unique_desc_ids, counts):
        num_progenitors[halo_ids == desc_id] = count
    return num_progenitors

def process_progenitors(
    halo_ids, halo_desc_ids, halo_mass, node_feats,
    num_max_prog=1, min_mass_ratio=0.01, return_prog_position=True
    ):
    """ Process the progenitors of each halo by sorting them by mass and removin
    any progenitors that are not the most massive or have a mass ratio less than
    min_mass_ratio. """

    def fun(index, halo_ids, halo_desc_ids, halo_mass, num_max_prog=1, min_mass_ratio=0.01, prog_pos=0):
        """ Recursively sort the progenitors of a halo by mass and return the sorted indices. """

        # Get the current halo ID and indices of its progenitors
        halo_id = halo_ids[index]
        halo_desc_indices = np.where(halo_desc_ids == halo_id)[0]

        # Start with the current halo index in the sorted list
        sorted_index = [index, ]
        prog_position = [prog_pos, ]

        if len(halo_desc_indices) == 0:
            return sorted_index, prog_position

        # Sort the progenitors by mass in descending order
        sort = np.argsort(halo_mass[halo_desc_indices])[::-1]
        sorted_desc_indices = halo_desc_indices[sort]
        halo_desc_mass = halo_mass[sorted_desc_indices]
        halo_desc_mass_ratio = halo_desc_mass / np.max(halo_desc_mass)

        mask = halo_desc_mass_ratio > min_mass_ratio
        sorted_desc_indices = sorted_desc_indices[mask][:num_max_prog]

        # Recursively process each progenitor and gather their sorted indices
        for i, progenitor_index in enumerate(sorted_desc_indices):
            s, p = fun(
                progenitor_index,
                halo_ids,
                halo_desc_ids,
                halo_mass,
                num_max_prog=num_max_prog,
                min_mass_ratio=min_mass_ratio,
                prog_pos=i
            )
            sorted_index.extend(s)
            prog_position.extend(p)

        return sorted_index, prog_position

    sorted_index, prog_position = fun(
        0, halo_ids, halo_desc_ids, halo_mass,
        num_max_prog=num_max_prog, min_mass_ratio=min_mass_ratio, prog_pos=0)
    new_node_feats = node_feats[sorted_index]
    new_halo_ids = halo_ids[sorted_index]
    new_halo_desc_ids = halo_desc_ids[sorted_index]
    prog_position = np.array(prog_position)

    if return_prog_position:
        return new_halo_ids, new_halo_desc_ids, new_node_feats, prog_position
    else:
        return new_halo_ids, new_halo_desc_ids, new_node_feats

