
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj

def find_path_from_root(edge_index, node):
    path = [node]
    # Traverse back to root
    while True:
        # Find predecessors
        preds = (edge_index[1] == node).nonzero(as_tuple=True)[0]
        if preds.nelement() == 0:
            break  # Reached a root node
        # Assuming single predecessor for simplicity
        node = edge_index[0][preds[0]]
        path.append(node)
    # Convert path to a LongTensor and reverse it
    path_tensor = torch.LongTensor(path).flip(dims=[0])
    return path_tensor

def find_ancestor_indices(edge_index, node):
    # Find indices where the selected node is the descendant halo
    ancestor_indices = (edge_index[0] == node).nonzero(as_tuple=True)[0]

    # Get the ancestors of the selected node
    # (i.e., the halos at the next snapshots that are connected to the
    # selected node)
    ancestors = edge_index[1][ancestor_indices]

    return ancestors

# def pad_sequences(sequences, max_len=None, padding_value=0):
#     if max_len is None:
#         max_len = max(len(seq) for seq in sequences)

#     padded_sequences = []
#     original_lengths = []
#     for seq in sequences:
#         original_lengths.append(len(seq))
#         padding_length = max_len - len(seq)
#         padded_seq = nn.functional.pad(seq, (0, 0, 0, padding_length), value=padding_value)
#         padded_sequences.append(padded_seq)

    # return torch.stack(padded_sequences), torch.tensor(original_lengths)

def pad_sequences(sequences, max_len=None, padding_value=0):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    seq_dim = sequences[0].dim()

    padded_sequences = []
    original_lengths = []
    for seq in sequences:
        original_lengths.append(len(seq))
        padding_length = max_len - len(seq)
        pad = [0] * (seq_dim * 2)
        pad[-1] = padding_length
        padded_seq = nn.functional.pad(seq, pad, value=padding_value)
        padded_sequences.append(padded_seq)

    return torch.stack(padded_sequences), torch.tensor(original_lengths)

def create_padding_mask(lengths, max_len, batch_first=False):
    """ Create a padding mask. """
    batch_size = lengths.size(0)

    # Generate a mask with shape [batch_size, seq_len]
    mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    if not batch_first:
        mask = mask.transpose(0, 1)
    return mask

def get_leaves(data):
    """
    Identify and return the leaf nodes in a graph.

    Parameters:
    data (torch_geometric.data.Data): The input graph data containing edge_index and num_nodes.

    Returns:
    torch.Tensor: A tensor containing the indices of the leaf nodes.
    """
    edge_index = data.edge_index.to(data.x.device)
    source_nodes = edge_index[0].unique()
    all_nodes = torch.arange(data.num_nodes, device=data.x.device)
    leaf_nodes = all_nodes[~torch.isin(all_nodes, source_nodes)]
    return leaf_nodes

def prepare_batch(batch, num_samples_per_graph=1, return_weights=False, all_nodes=False):
    """ Prepare a batch for training.

    Parameters
    ----------
    batch : torch_geometric.data.Batch
        The batch to prepare.
    num_samples_per_graph : int
        The number of samples to take per graph.
    return_weights : bool
        Whether to return the sample weights.
    all_nodes : bool
        If True, take all nodes instead of sampling. Overrides num_samples_per_graph.
    """

    features = []
    out_features = []
    weights = []

    for i in range(batch.num_graphs):
        if all_nodes:
            # select all nodes in the graph
            select_nodes = torch.arange(batch.ptr[i], batch.ptr[i + 1])
        else:
            # select a random node in the graph
            select_nodes = torch.randint(
                batch.ptr[i], batch.ptr[i + 1], (num_samples_per_graph,))

        for idx in select_nodes:
            # get the ancestors of the selected node and skip if there are none
            ancestors = find_ancestor_indices(batch.edge_index, idx)
            if ancestors.nelement() == 0:
                continue

            # create output feature vectors
            out_features.append(batch.x[ancestors])

            # create feature vectors
            # find the path to the root
            path = find_path_from_root(batch.edge_index, idx)

            features.append(batch.x[path])
            if return_weights:
                weights.append(batch.weight[i])

    # pad the features to the same length
    padded_features, lengths = pad_sequences(features)
    padded_out_features, out_lengths = pad_sequences(out_features)

    if return_weights:
        weights = torch.tensor(weights, dtype=torch.float32)
    else:
        weights = torch.ones(len(features), dtype=torch.float32)

    return (padded_features, lengths, padded_out_features, out_lengths, weights)

def prepare_batch_branch(
    batch, max_split, return_weights=False, all_nodes=False, use_desc_mass_ratio=False, use_prog_position=False):
    """ Prepare a batch for training.

    Parameters
    ----------
    batch : torch_geometric.data.Batch
        The batch to prepare.
    max_split : int
        The maximum number of progenitors to consider.
    return_weights : bool
        Whether to return the sample weights.
    all_nodes : bool
        If True, take all nodes instead of sampling. Overrides num_samples_per_graph.
    use_desc_mass_ratio: bool
        If True, out_features is a ratio of the descendants of the selected node.
        Assuming that index of the mass feature is 0.
    use_prog_position: bool
        If True, concatenate the position of the progenitors to the input time features.
    """
    features = []
    out_features = []
    num_progenitors = []
    weights = []
    max_num_prog_batch = 0

    for i in range(batch.num_graphs):
        graph = batch[i]
        leaves = get_leaves(graph)
        parents = graph.edge_index[0, torch.isin(graph.edge_index[1], leaves)]
        parents = parents[leaves-1 == parents]

        for idx in parents:
            path = find_path_from_root(graph.edge_index, idx)
            adj = to_dense_adj(graph.edge_index)[0]
            num_prog = torch.sum(adj[path], axis=1, dtype=torch.long)
            max_num_prog_batch = max(max_num_prog_batch, num_prog.max().item())

            if use_prog_position:
                feat_x = graph.x[path]
                feat_prog_pos = torch.nn.functional.one_hot(
                    graph.prog_pos[path].long(), num_classes=max_split).float()
                feat = torch.cat((feat_x, feat_prog_pos), dim=1)
            else:
                feat = graph.x[path]
            features.append(feat)
            num_progenitors.append(num_prog)

            out = torch.zeros((len(path), max_split, graph.x.size(1)), device=graph.x.device)
            for i, node in enumerate(path):
                temp = graph.x[adj[node].eq(1)]
                if use_desc_mass_ratio:
                    temp[:, 0] = temp[:, 0] - graph.x[node, 0]
                out[i][:num_prog[i]] += temp
            out_features.append(out)

            if return_weights:
                weights.append(graph.weight[path])

    padded_features, lengths = pad_sequences(features)
    padded_out_features, _ = pad_sequences(out_features)
    padded_out_features = padded_out_features[:, :, :max_num_prog_batch]
    num_progenitors, _ = pad_sequences(num_progenitors)

    if return_weights:
        weights = torch.tensor(weights, dtype=torch.float32)
    else:
        weights = torch.ones(len(features), dtype=torch.float32)

    return (padded_features, padded_out_features, num_progenitors, lengths, weights)
