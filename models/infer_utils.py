from typing import Dict, Optional, List, Any, Tuple

import time

import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm

from models import training_utils, models_utils

def check_mass_sort(tree):
    """ Check if the mass is sorted correctly within the tree """
    halo_idx, num_progs = torch.unique(tree.edge_index[0], return_counts=True)

    is_sorted = True
    for idx in halo_idx:
        progs_idx = tree.edge_index[1][tree.edge_index[0] == idx]
        progs_x = tree.x[progs_idx]

        # Check if progs_x[:, 0] is sorted in descending order
        is_sorted_prog = torch.all(progs_x[:, 0][:-1] >= progs_x[:, 0][1:])
        is_sorted = is_sorted & is_sorted_prog
        if not is_sorted_prog:
            print(f"Halo {idx} is not sorted correctly")
            print(tree.x[idx], progs_x, progs_idx)
    return is_sorted

def sort_tree(tree, sort_prop=0):
    def _sort_tree(tree, root_index=0):
        """ Sort the tree in a depth-first order """
        ordered_index = [root_index, ]
        prog_indices = tree.edge_index[1, tree.edge_index[0] == root_index]
        if len(prog_indices) == 0:
            return ordered_index
        prog_indices = prog_indices[
            torch.argsort(tree.x[prog_indices, sort_prop], descending=True)]
        for prog_index in prog_indices:
            ordered_index.extend(_sort_tree(tree, prog_index))
        return ordered_index

    ordered_index = _sort_tree(tree, torch.tensor(0, device=tree.x.device))
    ordered_index = torch.tensor(ordered_index, device=tree.x.device)
    ordered_index_list = ordered_index.tolist()

    new_tree = tree.clone()
    for key, value in tree.items():
        if key not in ['edge_index']:
            new_tree[key] = value[ordered_index]
    new_tree.edge_index = torch.stack([
        torch.tensor([ordered_index_list.index(i.item()) for i in tree.edge_index[0]]),
        torch.tensor([ordered_index_list.index(i.item()) for i in tree.edge_index[1]])
    ])
    perm = torch.argsort(new_tree.edge_index[1])
    new_tree.edge_index = new_tree.edge_index[:, perm]

    return new_tree

@torch.no_grad()
def generate_tree(
    model,
    x_root: torch.Tensor,
    t_out: torch.Tensor,
    norm_dict: Optional[Dict[str, torch.Tensor]] = None,
    n_max_iter: int = 1000,
    device: Optional[torch.device] = None,
):
    """
    Autoregressively generate a tree from the root node.

    Args:
        model: the autoregressive tree generation model
        x_root: the root node features, shape (n_features,)
        t_out: the output times, shape (n_times,)
        norm_dict: the normalization dictionary
        n_max_iter: the maximum number of iterations
        device: the device to run the model on

    Returns:
        tree: the generated tree
    """
    model.eval()
    use_desc_mass_ratio = model.training_args.use_desc_mass_ratio

    device = device or model.device
    if not isinstance(x_root, torch.Tensor):
        x_root = torch.tensor(x_root, device=device).float()
    else:
        x_root = x_root.to(device).float()
    if not isinstance(t_out, torch.Tensor):
        t_out = torch.tensor(t_out, device=device).float()
    else:
        t_out = t_out.to(device).float()

    # normalizing data
    if norm_dict is not None:
        x_loc = torch.tensor(norm_dict['x_loc'], dtype=torch.float32, device=device)
        x_scale = torch.tensor(norm_dict['x_scale'], dtype=torch.float32, device=device)
        t_loc = torch.tensor(norm_dict['t_loc'], dtype=torch.float32, device=device)
        t_scale = torch.tensor(norm_dict['t_scale'], dtype=torch.float32, device=device)
    else:
        x_loc = torch.zeros(x_root.size(0), device=device)
        x_scale = torch.ones(x_root.size(0), device=device)
        t_loc = torch.zeros(1, device=device)
        t_scale = torch.ones(1, device=device)
    x_root = (x_root - x_loc) / x_scale
    t_out = (t_out - t_loc) / t_scale

    # initialize the tree
    halo_t_index = [0]
    halo_index = [0]
    halo_remain_index = [0]  # index of the remining halos to be processed
    next_halo_index = 1
    edge_index= [[], []]  # keep edge index as a list to make appending easier
    halo_feats = torch.stack([x_root], dim=0)
    time_feats = torch.stack([t_out[0]], dim=0)
    prog_positions = torch.stack([torch.tensor([0, ], dtype=torch.float32, device=device)], dim=0)
    snap_indices = [0]

    # start generating the tree
    while (len(halo_remain_index) > 0) & (n_max_iter > 0):
        halo_curr_index = halo_remain_index.pop(0)
        halo_curr_t_index = halo_t_index.pop(0)
        if halo_curr_t_index == len(t_out) - 1:
            # reach the last snapshot
            continue

        # 0. get input sequence features and next time step
        path = training_utils.find_path_from_root(
            torch.tensor(edge_index, dtype=torch.long), halo_curr_index)

        # Processing Transformer input and time steps
        src = halo_feats[path].unsqueeze(0)  # add batch dim
        src_t = t_out[:halo_curr_t_index+1].unsqueeze(0)  # add batch dim
        tgt_t = t_out[halo_curr_t_index + 1].unsqueeze(0) # add batch dim

        # start generating the next halo
        # 1. pass the input sequence through the encoder
        x_enc = model.encoder(src, src_t, src_padding_mask=None)
        x_enc_reduced = models_utils.summarize_features(
            x_enc, reduction='last', padding_mask=None)

        # 2. pass the encoded sequence through the classifier
        # sample the number of progenitors from the predicted distribution
        x_class = x_enc_reduced + model.classifier_context_embed(tgt_t)
        x_class = model.classifier(x_class)
        y_class = torch.multinomial(x_class.softmax(dim=-1), 1)
        y_class_onehot = torch.nn.functional.one_hot(
            y_class, num_classes=model.num_classes).flows()
        n_prog = (y_class + 1).item()

        # 3. create the progenitors using the decoder and the flows
        tgt_in = torch.zeros((1, n_prog+1, x_root.size(0)), device=device)
        for iprog in range(n_prog):
            # pass the encoded sequence through the decoder
            if model.decoder_args.name == 'transformer':
                x_dec = model.decoder(
                    tgt_in[:, :-1],
                    memory=x_enc,
                    context=tgt_t,
                    tgt_padding_mask=None,
                    memory_padding_mask=None
                )
            elif model.decoder_args.name == 'gru':
                x_dec = model.decoder(
                    x=tgt_in,
                    t=tgt_t.unsqueeze(1).expand(-1, tgt_in_src.size(1), -1),
                    lengths=torch.tensor([iprog+1, ], dtype=torch.long)
                )

            # sample the next halo from the flows
            context = (
                x_dec
                + model.npe_context_embed(y_class_onehot).unsqueeze(1).expand_as(x_dec)
                + x_enc_reduced.unsqueeze(1).expand_as(x_dec)
            )
            context = context[:, -1]
            tgt_in[:, iprog + 1] = model.npe.flow(model.npe(context)).sample()

        prog_feats = tgt_in[0, 1:]
        if use_desc_mass_ratio:
            prog_feats[:, 0] = prog_feats[:, 0] + src[:, -1, 0]

        # print(time_feats.shape, halo_feats.shape, n_prog, tgt_in.shape, tgt_t.shape)
        halo_feats = torch.cat([halo_feats, prog_feats], dim=0)
        time_feats = torch.cat([time_feats, tgt_t.repeat(n_prog, 1)], dim=0)
        prog_positions = torch.cat(
            [prog_positions, torch.arange(n_prog).unsqueeze(1)], dim=0)
        snap_indices.extend([halo_curr_t_index + 1] * n_prog)

        # create halo index for the progenitors
        prog_index = [next_halo_index + i for i in range(n_prog)]
        halo_index = halo_index + prog_index
        halo_remain_index = halo_remain_index + prog_index
        halo_t_index = halo_t_index + [halo_curr_t_index + 1] * n_prog

        # create edge index for the progenitors
        edge_index[0] = edge_index[0] + [halo_curr_index] * n_prog
        edge_index[1] = edge_index[1] + prog_index
        next_halo_index += n_prog

        n_max_iter -= 1
        if n_max_iter == 0:
            print('Max number of iterations reached')

    # unnormalize the features
    halo_feats = halo_feats * x_scale + x_loc
    time_feats = time_feats * t_scale + t_loc
    return Data(
        x=torch.cat([halo_feats, time_feats], dim=1),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        prog_positions=prog_positions,
        snap_indices=torch.tensor(snap_indices, dtype=torch.long),
    )

def generate_forest(
    model,
    root_features: List[torch.Tensor],
    times_out: List[torch.Tensor],
    norm_dict: Optional[Dict[str, torch.Tensor]] = None,
    n_max_iter: int = 1000,
    device: Optional[torch.device] = None,
    snapshot_list: Optional[List[int]] = None,
    sort: bool = False,
    verbose: bool = False
):
    """ Generate multiple trees

    NOTE: the root_features should have the dimnesion of (batch_size, n_feat) instead
    of (n_feat,) even if the batch size is 1.
    """
    if len(root_features) != len(times_out):
        raise ValueError('root_features and times_out must have the same length')

    # generate trees for each root halo
    tree_list = []
    root_halo_range = range(len(root_features))

    loop = tqdm(
        root_halo_range, desc="Generating trees",
        miniters=len(root_features) // 100, disable=not verbose)

    time_start = time.time()
    for i in loop:
        tree = generate_tree(
            model,
            x_root=root_features[i],
            t_out=times_out[i],
            norm_dict=norm_dict,
            n_max_iter=n_max_iter,
            device=device
        )
        if snapshot_list is not None:
            tree.snap = snapshot_list[i][tree.snap_indices]
        if sort:
            tree = sort_tree(tree)
        # always return a CPU tensor for memory efficiency
        tree_list.append(tree.cpu())
    time_end = time.time()
    if verbose:
        print(f'Time elapsed: {time_end - time_start:.2f} s')

    return tree_list


# @torch.no_grad()
# def generate_mb_batch(
#     model,
#     x_root: torch.Tensor,
#     t_out: torch.Tensor,
#     norm_dict: Dict[str, Any],
#     device: torch.device,
#     verbose: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor]:

#     device = device or model.device
#     model.eval()
#     model.to(device)
#     use_desc_mass_ratio = model.training_args.use_desc_mass_ratio

#     if not isinstance(x_root, torch.Tensor):
#         x_root = torch.tensor(x_root, device=device).float()
#     else:
#         x_root = x_root.to(device).float()
#     if not isinstance(t_out, torch.Tensor):
#         t_out = torch.tensor(t_out, device=device).float()
#     else:
#         t_out = t_out.to(device).float()

#     # normalizing data
#     if norm_dict is not None:
#         x_loc = torch.tensor(norm_dict['x_loc'], dtype=torch.float32, device=device)
#         x_scale = torch.tensor(norm_dict['x_scale'], dtype=torch.float32, device=device)
#         t_loc = torch.tensor(norm_dict['t_loc'], dtype=torch.float32, device=device)
#         t_scale = torch.tensor(norm_dict['t_scale'], dtype=torch.float32, device=device)
#     else:
#         x_loc = torch.zeros(x_root.size(0), device=device)
#         x_scale = torch.ones(x_root.size(0), device=device)
#         t_loc = torch.zeros(1, device=device)
#         t_scale = torch.ones(1, device=device)
#     x_root = (x_root - x_loc) / x_scale
#     t_out = (t_out - t_loc) / t_scale

#     n_t = t_out.size(1)
#     halo_feats = torch.zeros(
#         (x_root.size(0), n_t, x_root.size(1)), dtype=torch.float32, device=device)
#     halo_feats[:, 0] = x_root
#     class_output = torch.zeros(
#         (x_root.size(0), n_t, model.num_classes), dtype=torch.float32, device=device)

#     for i in range(n_t-1):
#         src = halo_feats[:, :i + 1]
#         tgt_t = t_out[:, i + 1]
#         # if model.concat_time:
#             # src_t = torch.cat([t_out[:, :i + 1], t_out[:, 1:i+2]], dim=-1)
#         # else:
#         src_t = t_out[:, :i + 1]

#         # pss the source sequence through the encoder
#         if model.encoder_args.name == 'transformer':
#             x_enc = model.encoder(src, src_t, src_padding_mask=None)
#             x_enc_reduced = models_utils.summarize_features(
#                 x_enc, reduction='last', padding_mask=None)
#         elif model.encoder_args.name == 'gru':
#             x_enc = model.encoder(
#                 src,
#                 src_t,
#                 lengths=torch.tensor([i + 1, ], dtype=torch.long).expand(src.size(0))
#             )
#             x_enc_reduced = models_utils.summarize_features(
#                 x_enc, reduction='last', padding_mask=None)

#         # pass the encoded sequence through the decoder
#         if model.use_sos_embedding:
#             tgt_in = model.sos_embedding(x_enc_reduced).unsqueeze(1)
#         else:
#             tgt_in = torch.zeros((x_root.size(0), 1, x_root.size(1)), device=device)

#         if model.decoder_args.name == 'transformer':
#             x_dec = model.decoder(
#                 tgt_in,
#                 memory=x_enc_reduced.unsqueeze(1),
#                 context=tgt_t,
#                 tgt_padding_mask=None,
#                 memory_padding_mask=None
#             )
#         elif model.decoder_args.name == 'gru':
#             x_dec = model.decoder(
#                 x=tgt_in,
#                 t=tgt_t.unsqueeze(1).expand(-1, tgt_in.size(1), -1),
#                 lengths=torch.ones((1, ), dtype=torch.long)
#             )
#         # sample the next halo from the flows
#         if model.concat_npe_context:
#             context_flows = torch.cat([
#                 x_dec,
#                 x_enc_reduced.unsqueeze(1).expand(1, x_dec.size(1), 1)
#             ], dim=-1)
#         else:
#             context_flows = x_dec + x_enc_reduced.unsqueeze(1).expand(-1, x_dec.size(1), -1)
#         context_flows = context_flows[:, -1]

#         prog_feat = model.npe.flow(context_flows).sample()
#         if use_desc_mass_ratio:
#             prog_feat[:, 0] = prog_feat[:, 0] + halo_feats[:, i, 0]
#         halo_feats[:, i+1] = prog_feat

#         # calculate and return x class
#         x_class = x_enc_reduced + model.classifier_context_embed(tgt_t)
#         x_class = model.classifier(x_class)
#         class_output[:, i+1] += x_class

#     # unnormalize the features
#     halo_feats = halo_feats * x_scale + x_loc
#     t_out = t_out * t_scale + t_loc
#     halo_feats = torch.cat([halo_feats, t_out], dim=-1)

#     return halo_feats, class_output

