from typing import Dict, Optional, List, Any, Tuple

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
def _generate_next_progenitors(model, loader, device):
    """ Return Xprog and Nprog """

    Xprog_all = []
    Nprog_all = []
    for batch in loader:
        Hhist_batch = batch[0].to(device)
        Zhist_batch = batch[1].to(device)
        z_prog_batch = batch[2].to(device)
        batch_size = Hhist_batch.size(0)

        # 1. pass through the encoder
        x_enc = model.encoder(Hhist_batch, Zhist_batch)

        # 2. pass through classifier and compute the multinomial distribution
        x_enc_reduced = models_utils.summarize_features(
            x_enc, reduction='last', padding_mask=None)
        x_class = x_enc_reduced + model.classifier_context_embed(z_prog_batch)
        x_class = model.classifier(x_class)
        y_class = torch.multinomial(x_class.softmax(dim=-1), 1)
        y_class_onehot = torch.nn.functional.one_hot(
            y_class, num_classes=model.num_classes).float()
        y_class_context = model.npe_context_embed(y_class_onehot).squeeze(1)
        Nprog = (y_class + 1)

        # 3. create the progenitors using the decoder and the flows
        # always generate the maximum number of progenitors
        Xprog = torch.zeros((batch_size, model.num_classes + 1, Hhist_batch.size(-1)), device=device)
        for i in range(model.num_classes):
            if model.decoder_args.name == 'transformer':
                x_dec = model.decoder(
                    Xprog[:, :i+1],
                    memory=x_enc,
                    context=z_prog_batch,
                    tgt_padding_mask=None,
                    memory_padding_mask=None
                )
            elif model.decoder_args.name == 'gru':
                x_dec = model.decoder(
                    x=Xprog[:, :i+1],
                    t=z_prog_batch.unsqueeze(1).expand(-1, i+1, -1),
                )
            x_dec = x_dec[:, -1]

            pos = torch.full((batch_size, ), i, dtype=torch.long, device=device)
            pos = torch.nn.functional.one_hot(pos, model.num_classes).float()
            pos = pos.to(device)
            pos_context = model.npe_position_embed(pos)
            context = x_dec + y_class_context + x_enc_reduced + pos_context
            prog_feats = model.npe.flow(model.npe(context)).sample()
            # prog_feats = torch.clamp(prog_feats, max=1.0)
            Xprog[:, i+1] = prog_feats

        # store the halos
        Xprog_all.append(Xprog[:, 1:])
        Nprog_all.append(Nprog)
    Xprog_all = torch.cat(Xprog_all, dim=0)
    Nprog_all = torch.cat(Nprog_all, dim=0)

    return Xprog_all, Nprog_all

@torch.no_grad()
def _generate_all_progenitors(
    model,
    x0: torch.Tensor,
    Zhist: torch.Tensor,
    norm_dict: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    """ Generate merger trees with batching """
    model.eval()
    use_desc_mass_ratio = model.training_args.use_desc_mass_ratio

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = x0.to(device).float()
    Zhist = Zhist.to(device).float()

    # normalizing data
    if norm_dict is not None:
        x_loc = torch.tensor(norm_dict['x_loc'], dtype=torch.float32, device=device)
        x_scale = torch.tensor(norm_dict['x_scale'], dtype=torch.float32, device=device)
        t_loc = torch.tensor(norm_dict['t_loc'], dtype=torch.float32, device=device)
        t_scale = torch.tensor(norm_dict['t_scale'], dtype=torch.float32, device=device)
    else:
        x_loc = torch.zeros(x0.size(0), device=device)
        x_scale = torch.ones(x0.size(0), device=device)
        t_loc = torch.zeros(1, device=device)
        t_scale = torch.ones(1, device=device)
    x0 = (x0 - x_loc) / x_scale
    Zhist = (Zhist - t_loc) / t_scale

    # start generating the trees snapshot-by-snapshot
    Ntree = len(x0)
    Nz = len(Zhist)

    # initialize the halo and time features with only the root
    halo_feats = [x0.cpu()]
    time_feats = [Zhist[:1].expand(Ntree, 1).cpu()]
    edge_index = [[], []]
    snap_index = [0] * Ntree
    tree_index = list(range(Ntree))

    Hhist_desc_next = x0.unsqueeze(1)
    halo_index_next = Ntree
    desc_index_curr = list(range(Ntree))

    for i in range(Nz-1):
        Hhist_desc = Hhist_desc_next
        Zhist_desc = Zhist[:i+1].unsqueeze(0).expand(len(Hhist_desc), i+1, 1)
        z_prog = Zhist[i+1].unsqueeze(0).expand(len(Hhist_desc), 1)

        loader = DataLoader(
            TensorDataset(Hhist_desc, Zhist_desc, z_prog), batch_size=batch_size,
            shuffle=False)

        if verbose:
            print('Generating progenitors for snapshot {}/{}'.format(i+1, Nz-1))
            loader = tqdm(loader)
        Xprog, Nprog = _generate_next_progenitors(model, loader, device)

        # construct the descendant history list for the next snapshot
        Hhist_desc_next = []
        desc_index_next = []
        for j in range(len(Xprog)):
            for k in range(Nprog[j]):
                Xprog_ = Xprog[j, k].unsqueeze(0)
                Hhist_desc_next.append(torch.cat([Hhist_desc[j], Xprog_], dim=0))

                # store the progenitor information
                halo_feats.append(Xprog_.cpu())
                time_feats.append(Zhist[i+1].unsqueeze(0).cpu())
                edge_index[0].append(desc_index_curr[j])
                edge_index[1].append(halo_index_next)
                tree_index.append(tree_index[desc_index_curr[j]])   # same tree index with the parent
                snap_index.append(i+1)
                desc_index_next.append(halo_index_next)
                halo_index_next += 1

        Hhist_desc_next = torch.stack(Hhist_desc_next, dim=0)
        desc_index_curr = desc_index_next

    halo_feats = torch.cat(halo_feats, dim=0)
    time_feats = torch.cat(time_feats, dim=0)
    halo_feats = halo_feats * x_scale.cpu() + x_loc.cpu()
    time_feats = time_feats * t_scale.cpu() + t_loc.cpu()
    halo_index = torch.arange(len(halo_feats), dtype=torch.long)
    tree_index = torch.tensor(tree_index, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    snap_index = torch.tensor(snap_index, dtype=torch.long)

    return halo_feats, time_feats, halo_index, tree_index, edge_index, snap_index

@torch.no_grad()
def generate_forest(
    model,
    x0: torch.Tensor,
    Zhist: torch.Tensor,
    norm_dict: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
    snapshot_list: Optional[List[int]] = None,
    sort: bool = False,
    verbose: bool = False,
):
    """
    Generate merger trees with batching. Including multiple steps:
    1. generate all progenitor for all root halos
    2. reconstruct the trees by connecting the halos based on their tree index
    """
    t1 = time.time()

    # 1. Generate all progenitors for all root halos
    Ntree = len(x0)
    halo_feats, time_feats, halo_index, tree_index, edge_index, snap_index = _generate_all_progenitors(
        model, x0, Zhist, norm_dict=norm_dict, batch_size=batch_size, device=device, verbose=verbose,
    )

    t2 = time.time()

    # 2. Reconstruct the tree
    forest = []   # list of trees
    if verbose:
        print("Reconstructing the trees...")
        unique_tree_index = tqdm(range(Ntree))
    else:
        unique_tree_index = range(Ntree)
    for i in unique_tree_index:
        halo_feats_t = halo_feats[tree_index == i]
        time_feats_t = time_feats[tree_index == i]
        snap_index_t = snap_index[tree_index == i]

        # remap the global halo index to the local halo index (range from 0 to Nhalo)
        halo_index_t = halo_index[tree_index == i]
        edge_index_t = edge_index[:, (tree_index[edge_index[0]] == i) & (tree_index[edge_index[1]] == i)]
        edge_index_t_remap = torch.searchsorted(halo_index_t, edge_index_t)

        # create the tree
        tree = Data(
            x=torch.cat([halo_feats_t, time_feats_t], dim=1),
            edge_index=edge_index_t_remap,
            snap_indices=snap_index_t,
        )
        if snapshot_list is not None:
            tree.snap = snapshot_list[tree.snap_indices]
        if sort:
            tree = sort_tree(tree)
        forest.append(tree)
    t3 = time.time()

    if verbose:
        print('Time elapsed for generating progenitors:', t2 - t1)
        print('Time elapsed for reconstructing the trees:', t3 - t2)
        print('Total time elapsed:', t3 - t1)

    return forest
