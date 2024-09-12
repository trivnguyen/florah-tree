from typing import Dict, Optional, List, Any, Tuple

import time

import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm

from models import training_utils, models_utils

@torch.no_grad()
def generate_tree(
    model,
    x_root: torch.Tensor,
    t_out: torch.Tensor,
    norm_dict: Optional[Dict[str, torch.Tensor]] = None,
    n_max_iter: int = 1000,
    device: Optional[torch.device] = None
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
        src = halo_feats[path].unsqueeze(0)  # add batch dim
        src_t = t_out[:halo_curr_t_index+1].unsqueeze(0)  # add batch dim
        tgt_t = t_out[halo_curr_t_index + 1].unsqueeze(0).unsqueeze(0)  # add batch and feature dim

        # start generating the next halo
        # 1. pass the input sequence through the encoder
        x_enc = model.encoder(src, src_t, src_padding_mask=None)

        # 2. pass the encoded sequence through the classifier
        # sample the number of progenitors from the predicted distribution
        x_enc_reduced = models_utils.summarize_features(
            x_enc, reduction='mean', padding_mask=None)
        x_class = x_enc_reduced + model.classifier_context_embed(tgt_t)
        x_class = model.classifier(x_class)
        n_prog = torch.multinomial(x_class.softmax(dim=1), 1) + 1
        n_prog = n_prog.item()

        # 3. create the progenitors using the decoder and the flows
        tgt_in = torch.zeros((1, n_prog+1, x_root.size(0)), device=device)
        for iprog in range(n_prog):
            # pass the encoded sequence through the decoder
            x_dec = model.decoder(
                tgt_in[:, :-1],
                memory=x_enc,
                context=tgt_t,
                tgt_padding_mask=None,
                memory_padding_mask=None
            )
            # sample the next halo from the flows
            context_flows = x_dec + x_enc_reduced.unsqueeze(1).expand(-1, x_dec.size(1), -1)
            context_flows = context_flows[:, -1]
            tgt_in[:, iprog + 1] = model.npe.flow(model.npe(context_flows)).sample()

        halo_feats = torch.cat([halo_feats, tgt_in[0, 1:]], dim=0)
        time_feats = torch.cat([time_feats, tgt_t[0].repeat(n_prog)], dim=0)

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
        x=torch.cat([halo_feats, time_feats.unsqueeze(1)], dim=1),
        edge_index=torch.tensor(edge_index, dtype=torch.long)
    )

@torch.no_grad()
def generate_mb_batch(
    model,
    x_root: torch.Tensor,
    t_out: torch.Tensor,
    norm_dict: Dict[str, Any],
    device: torch.device,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = device or model.device
    model.eval()
    model.to(device)
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

    n_t = t_out.size(1)
    halo_feats = torch.zeros(
        (x_root.size(0), n_t, x_root.size(1)), dtype=torch.float32, device=device)
    halo_feats[:, 0] = x_root
    class_output = torch.zeros(
        (x_root.size(0), n_t, model.num_classes), dtype=torch.float32, device=device)

    for i in range(n_t-1):
        src = halo_feats[:, :i + 1]
        src_t = t_out[:, :i + 1]
        tgt_t = t_out[:, i + 1]

        # pss the source sequence through the encoder
        x_enc = model.encoder(src, src_t, src_padding_mask=None)
        x_enc_reduced = models_utils.summarize_features(
            x_enc, reduction='mean', padding_mask=None)

        # pass the encoded sequence through the decoder
        tgt_in = torch.zeros((x_root.size(0), 1, x_root.size(1)), device=device)
        x_dec = model.decoder(
            tgt_in,
            memory=x_enc,
            context=tgt_t,
            tgt_padding_mask=None,
            memory_padding_mask=None
        )
        # sample the next halo from the flows
        context_flows = x_dec + x_enc_reduced.unsqueeze(1).expand(-1, x_dec.size(1), -1)
        context_flows = context_flows[:, -1]
        halo_feats[:, i+1] = model.npe.flow(model.npe(context_flows)).sample()

        # calculate and return x class
        x_class = x_enc_reduced + model.classifier_context_embed(tgt_t)
        x_class = model.classifier(x_class)
        class_output[:, i+1] += x_class

    # unnormalize the features
    halo_feats = halo_feats * x_scale + x_loc
    t_out = t_out * t_scale + t_loc
    halo_feats = torch.cat([halo_feats, t_out], dim=-1)

    return halo_feats, class_output


def generate_forest(
    model,
    root_features: List[torch.Tensor],
    times_out: List[torch.Tensor],
    norm_dict: Optional[Dict[str, torch.Tensor]] = None,
    n_max_iter: int = 1000,
    device: Optional[torch.device] = None,
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

    if verbose:
        loop = tqdm(
            root_halo_range, desc="Generating trees",
            miniters=len(root_features) // 100)
    else:
        loop = root_halo_range

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
        # always return a CPU tensor for memory efficiency
        tree_list.append(tree.cpu())
    time_end = time.time()
    if verbose:
        print(f'Time elapsed: {time_end - time_start:.2f} s')

    return tree_list
