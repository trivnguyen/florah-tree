
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
    source_nodes = data.edge_index[0]
    target_nodes = data.edge_index[1]
    return target_nodes[torch.where(~torch.isin(target_nodes, source_nodes))[0]]

def prepare_batch_branch(
    batch, max_split, return_weights=False, num_branches_per_tree=None,
    use_desc_mass_ratio=False, use_prog_position=False, use_leaves=False
):
    """ Prepare a batch for training.

    Parameters
    ----------
    batch : torch_geometric.data.Batch
        The batch to prepare.
    max_split : int
        The maximum number of progenitors to consider.
    return_weights : bool
        Whether to return the sample weights.
    num_branches_per_tree: int
        The number of branches to consider for each tree. If None, all branches are considered.
    use_desc_mass_ratio: bool
        If True, out_features is a ratio of the descendants of the selected node.
        Assuming that index of the mass feature is 0.
    use_prog_position: bool
        If True, concatenate the position of the progenitors to the input time features.
    use_leaves: bool
        If True, include the leaves as descendants. Required if num_prog=0 is allowed in the classifier.
    """
    features = []
    out_features = []
    num_progenitors = []
    weights = []
    max_num_prog_batch = 0

    for i in range(batch.num_graphs):
        graph = batch[i].cpu()  # convert to CPU to avoid memory issues
        graph_nprog = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)

        leaves = get_leaves(graph)
        if not use_leaves:
            parents = graph.edge_index[0, torch.isin(graph.edge_index[1], leaves)]
            parents = parents[leaves-1 == parents]
        else:
            parents = leaves

        if num_branches_per_tree is not None:
            # select a random subset of parents
            parents = parents[torch.randperm(len(parents))[:num_branches_per_tree]]

        for idx in parents:
            path = find_path_from_root(graph.edge_index, idx)
            num_prog = graph_nprog[path]
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
                temp = graph.x[graph.edge_index[1][graph.edge_index[0]==node]]
                if use_desc_mass_ratio:
                    temp[:, 0] = temp[:, 0] - graph.x[node, 0]
                out[i][:num_prog[i]] += temp[:max_split]
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
