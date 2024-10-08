
import torch
import torch.nn as nn


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

    padded_sequences = []
    original_lengths = []
    for seq in sequences:
        original_lengths.append(len(seq))
        padding_length = max_len - len(seq)
        padded_seq = nn.functional.pad(seq, (0, 0, 0, padding_length), value=padding_value)
        padded_sequences.append(padded_seq)

    return torch.stack(padded_sequences), torch.tensor(original_lengths)

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


def create_padding_mask(lengths, max_len, batch_first=False):
    """ Create a padding mask. """
    batch_size = lengths.size(0)

    # Generate a mask with shape [batch_size, seq_len]
    mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    if not batch_first:
        mask = mask.transpose(0, 1)
    return mask