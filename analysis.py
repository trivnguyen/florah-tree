

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

def remove_anc(halo_ids, halo_desc_ids, halo_mass, node_feats, num_max_anc=1):
    """ Remove enforce the maxmium number of ancestors. If more than num_max_anc
    halos have the same descendant, only keep the num_max_anc most massive ones.

    Note that this code assumes the halos are sorted by snapshot number starting
    from teh root halo.
    """
    if num_max_anc < 1:
        raise ValueError("num_max_anc must be >= 1.")

    # identify the halos with the same descendant
    unique_desc_ids, counts = np.unique(halo_desc_ids, return_counts=True)
    bad_indices = []
    for desc_id, count in zip(unique_desc_ids, counts):
        if count <= num_max_anc:
            continue
        # find the indices of the halos with this desc_id
        indices = np.where(halo_desc_ids == desc_id)[0]

        # sort the indices by mass and keep the num_max_anc most massive ones
        mass_indices = np.argsort(halo_mass[indices])[::-1]
        bad_indices.append(indices[mass_indices[num_max_anc:]])

    if len(bad_indices) == 0:
        return halo_ids, halo_desc_ids, node_feats
    else:
        bad_indices = np.concatenate(bad_indices)

    # iterate over all halos and add only good halos
    # good halos are those that are not in bad_indices and have a valid desc_id
    new_node_feats = []
    new_halo_ids = []
    new_halo_desc_ids = []
    for i in range(len(halo_ids)):
        accept = (i not in bad_indices) and np.isin(halo_desc_ids[i], new_halo_ids)
        accept = accept or (halo_desc_ids[i] == -1)  # always keep the root halo (desc_id = -1)
        if accept:
            new_node_feats.append(node_feats[i])
            new_halo_ids.append(halo_ids[i])
            new_halo_desc_ids.append(halo_desc_ids[i])
    new_node_feats = np.stack(new_node_feats, axis=0)
    new_halo_ids = np.array(new_halo_ids)
    new_halo_desc_ids = np.array(new_halo_desc_ids)

    return new_halo_ids, new_halo_desc_ids, new_node_feats

def calc_num_ancestors(halo_ids, halo_desc_ids):
    """ Calculate the number of ancestors for each halo. """
    unique_desc_ids, counts = np.unique(halo_desc_ids, return_counts=True)
    num_ancestors = np.zeros(len(halo_desc_ids), dtype=np.int32)
    for desc_id, count in zip(unique_desc_ids, counts):
        num_ancestors[halo_ids == desc_id] = count
    return num_ancestors
