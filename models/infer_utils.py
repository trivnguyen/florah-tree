
import time

import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm

from . import training_utils


def generate_tree_A(
    classifier, regressor, root_halo_feat, t_out, classifier_norm_dict=None,
    regressor_norm_dict=None, n_max_iter=1000, device='cpu'):
    """Generate a tree using a separate classifier and regressor.

    Parameters
    ----------
    classifier : nn.Module
        The classifier model.
    regressor : nn.Module
        The regressor model.
    root_halo_feat : torch.Tensor
        The feature vector of the root halo. The shape should be (n_feat,).
        Not including the time dimension.
    t_out : torch.Tensor
        The output time vector. The shape should be (n_t,).
    classifier_norm_dict : dict
        The normalization dictionary for the classifier.
    regressor_norm_dict : dict
        The normalization dictionary for the regressor.
    n_max_iter : int, optional
        The maximum number of iterations. The default is 1000.
    device : str, optional
        The device to use. The default is 'cpu'.

    Returns
    -------
    tree : torch_geometric.data.Data
        The tree data with the following attributes:
            x : torch.Tensor
                The feature matrix. The shape should be (n_nodes, n_feat+1).
                Including the time dimension.
            edge_index : torch.Tensor
                The edge index matrix. The shape should be (2, n_edges).
    """
    num_feat = len(root_halo_feat)

    # convert to tensor if not already and move to dvice
    if not isinstance(t_out, torch.Tensor):
        t_out = torch.tensor(t_out, dtype=torch.float32)
    if not isinstance(root_halo_feat, torch.Tensor):
        root_halo_feat = torch.tensor(root_halo_feat, dtype=torch.float32)
    t_out = t_out.to(device)
    root_halo_feat = root_halo_feat.to(device)
    root_halo_feat = torch.cat([root_halo_feat, t_out[0].unsqueeze(0)])

    # normalization
    if classifier_norm_dict is not None:
        classifier_x_loc = torch.tensor(
            classifier_norm_dict['x_loc'], dtype=torch.float32, device=device)
        classifier_x_scale = torch.tensor(
            classifier_norm_dict['x_scale'], dtype=torch.float32, device=device)
    else:
        classifier_x_loc = torch.zeros(num_feat+1, dtype=torch.float32, device=device)
        classifier_x_scale = torch.ones(num_feat+1, dtype=torch.float32, device=device)

    if regressor_norm_dict is not None:
        regressor_x_loc = torch.tensor(
            regressor_norm_dict['x_loc'], dtype=torch.float32, device=device)
        regressor_x_scale = torch.tensor(
            regressor_norm_dict['x_scale'], dtype=torch.float32, device=device)
    else:
        regressor_x_loc = torch.zeros(num_feat+1, dtype=torch.float32, device=device)
        regressor_x_scale = torch.ones(num_feat+1, dtype=torch.float32, device=device)

    # initialize the tree
    halo_t_index = [0]
    halo_index = [0]
    halo_remain_index = [0]  # index of the remining halos to be processed
    next_halo_index = 1
    halo_feats = torch.stack([root_halo_feat], dim=0)
    edge_index= [[], []]  # keep edge index as a list to make appending easier
    next_halo_index = 1

    regressor.eval()
    classifier.eval()
    with torch.no_grad():
        while (len(halo_remain_index) > 0) & (n_max_iter > 0):
            halo_curr_index = halo_remain_index.pop(0)
            halo_curr_t_index = halo_t_index.pop(0)
            if halo_curr_t_index == len(t_out) - 1:
                # reach the last snapshot
                continue
            t_next = t_out[halo_curr_t_index + 1]
            t_next = t_next.unsqueeze(0).unsqueeze(0)  # add batch and feature dim
            path = training_utils.find_path_from_root(
                    torch.tensor(edge_index, dtype=torch.long),
                    halo_curr_index)

            # Classifier to predict the number of progenitors
            # extract features, and apply proper normalization
            x_feat_class = halo_feats[path]
            x_feat_class = (x_feat_class - classifier_x_loc) / classifier_x_scale
            t_next_class = (t_next - classifier_x_loc[-1]) / classifier_x_scale[-1]

            # pass through the transformer and the time projection layer
            x_feat_class = classifier.featurizer(x_feat_class.unsqueeze(0)) # add batch dim
            t_proj_class = classifier.time_proj_layer(t_next_class)
            # get the number of progenitors
            x_class = torch.cat((x_feat_class, t_proj_class), dim=1)
            yhat = classifier.classifier(x_class).softmax(dim=1)
            num_progs = torch.multinomial(yhat, 1) + 1
            num_progs = num_progs.item()

            # Regressor to predict the progenitor features
            # extract features, and apply proper normalization
            x_feat_regress = halo_feats[path]
            x_feat_regress = (x_feat_regress - regressor_x_loc) / regressor_x_scale
            t_next_regress = (t_next - regressor_x_loc[-1]) / regressor_x_scale[-1]
            # pass through the transformer and the time projection layer
            x_feat_regress = regressor.featurizer(x_feat_regress.unsqueeze(0)) # add batch dim
            t_proj_regress = regressor.time_proj_layer(t_next_regress)

            # list of progenitors, including the zeros starting token
            x_progs = torch.zeros((1, num_progs + 1, num_feat), device=device)
            t_proj_progs = t_proj_regress.unsqueeze(1).repeat(1, num_progs + 1, 1)

            for i_prog in range(num_progs):
                n_prog_curr = i_prog + 1  # number of progenitors at current step

                lengths = torch.tensor([n_prog_curr], dtype=torch.long)
                f_proj = regressor.feat_proj_layer(x_progs)

                # run the RNN to extract the context
                x_rnn = torch.cat([f_proj, t_proj_progs], dim=-1)
                x_rnn = regressor.rnn(x_rnn, lengths=lengths)

                # # sample from the flow
                flow_context = torch.cat(
                    [x_rnn, x_feat_regress.unsqueeze(1).repeat(1, n_prog_curr, 1)], dim=-1)
                flow_context = flow_context[:, -1]  # only take the last time step
                x_prog_curr = regressor.flows.sample(1, context=flow_context)

                # append to the list of progenitors
                x_progs[:, i_prog + 1] = x_prog_curr
            x_progs = x_progs[:, 1:]  # remove the zeros starting token]

            # add the progenitors to the list of halos
            x_progs = torch.cat([x_progs, t_next_regress.repeat(1, num_progs, 1)], dim=-1)
            x_progs = x_progs.squeeze(0)  # remove the batch dim
            x_progs = x_progs * regressor_x_scale + regressor_x_loc  # unnormalize
            halo_feats = torch.cat([halo_feats, x_progs], dim=0)

            # create halo index for the progneitors
            prog_index = [next_halo_index + i for i in range(num_progs)]
            halo_index = halo_index + prog_index
            halo_remain_index = halo_remain_index + prog_index
            halo_t_index = halo_t_index + [halo_curr_t_index + 1] * num_progs

            # create edge index for the progenitors
            edge_index[0] = edge_index[0] + [halo_curr_index] * num_progs
            edge_index[1] = edge_index[1] + prog_index
            next_halo_index += num_progs

            n_max_iter -= 1
            if n_max_iter == 0:
                print('Max number of iterations reached')

    tree = Data(x=halo_feats, edge_index=torch.tensor(edge_index, dtype=torch.long))
    return tree


def generate_tree_B(
    generator, root_halo_feat, t_out, n_max_iter=1000, norm_dict=None,
    device='cpu'):
    """Generate a tree using a tree generator model.

    Parameters
    ----------
    generator : torch.nn.Module
        The tree generator model.
    root_halo_feat : torch.Tensor
        The feature vector of the root halo. The shape should be (n_feat,).
        Not including the time dimension.
    t_out : torch.Tensor
        The output time vector. The shape should be (n_t,).
    norm_dict : dict, optional
        The normalization dictionary. The default is None.
    n_max_iter : int, optional
        The maximum number of iterations. The default is 1000.
    device : str, optional
        The device to use. The default is 'cpu'.

    Returns
    -------
    tree : torch_geometric.data.Data
        The tree data with the following attributes:
            x : torch.Tensor
                The feature matrix. The shape should be (n_nodes, n_feat+1).
                Including the time dimension.
            edge_index : torch.Tensor
                The edge index matrix. The shape should be (2, n_edges).
    """
    num_feat = len(root_halo_feat)

    # convert to tensors and move to device
    if not isinstance(t_out, torch.Tensor):
        t_out = torch.tensor(t_out, dtype=torch.float32)
    t_out = t_out.to(device)
    if not isinstance(root_halo_feat, torch.Tensor):
        root_halo_feat = torch.tensor(root_halo_feat, dtype=torch.float32)
    root_halo_feat = root_halo_feat.to(device)
    root_halo_feat = torch.cat([root_halo_feat, t_out[0].unsqueeze(0)])

    # normalization
    if norm_dict is not None:
        x_loc = torch.tensor(
            norm_dict['x_loc'], dtype=torch.float32, device=device)
        x_scale = torch.tensor(
            norm_dict['x_scale'], dtype=torch.float32, device=device)
    else:
        x_loc = torch.zeros(num_feat + 1, dtype=torch.float32, device=device)
        x_scale = torch.ones(num_feat + 1, dtype=torch.float32, device=device)

    # initialize the tree
    halo_t_index = [0]
    halo_index = [0]
    halo_remain_index = [0]  # index of the remining halos to be processed
    next_halo_index = 1
    halo_feats = torch.stack([root_halo_feat], dim=0)
    edge_index= [[], []]  # keep edge index as a list to make appending easier
    next_halo_index = 1

    generator.eval()
    with torch.no_grad():
        while (len(halo_remain_index) > 0) & (n_max_iter > 0):
            halo_curr_index = halo_remain_index.pop(0)
            halo_curr_t_index = halo_t_index.pop(0)
            if halo_curr_t_index == len(t_out) - 1:
                # reach the last snapshot
                continue
            t_next = t_out[halo_curr_t_index + 1]
            t_next = t_next.unsqueeze(0).unsqueeze(0)  # add batch and feature dim
            t_next = (t_next - x_loc[-1]) / x_scale[-1]

            # input features
            path = training_utils.find_path_from_root(
                torch.tensor(edge_index, dtype=torch.long),
                halo_curr_index)
            x_feat = halo_feats[path]
            x_feat = (x_feat - x_loc) / x_scale
            x_feat = generator.featurizer(x_feat.unsqueeze(0)) # add batch dim
            t_proj = generator.time_proj_layer(t_next)

            # randomly sample the number of progenitors
            x_classifier = torch.cat((x_feat, t_proj), dim=1)
            yhat = generator.classifier(x_classifier).softmax(dim=1)
            num_progs = torch.multinomial(yhat, 1) + 1
            num_progs = num_progs.item()

            # list of progenitors, including the zeros starting token
            x_progs = torch.zeros((1, num_progs + 1, num_feat), device=device)
            t_proj_progs = t_proj.unsqueeze(1).repeat(1, num_progs + 1, 1)

            for i_prog in range(num_progs):
                n_prog_curr = i_prog + 1  # number of progenitors at current step

                lengths = torch.tensor([n_prog_curr], dtype=torch.long)
                f_proj = generator.feat_proj_layer(x_progs)

                # run the RNN to extract the context
                x_rnn = torch.cat([f_proj, t_proj_progs], dim=-1)
                x_rnn = generator.rnn(x_rnn, lengths=lengths)

                # # sample from the flow
                flow_context = torch.cat(
                    [x_rnn, x_feat.unsqueeze(1).repeat(1, n_prog_curr, 1)], dim=-1)
                flow_context = flow_context[:, -1]  # only take the last time step
                x_prog_curr = generator.flows.sample(1, context=flow_context)

                # append to the list of progenitors
                x_progs[:, i_prog + 1] = x_prog_curr
            x_progs = x_progs[:, 1:]  # remove the zeros starting token]

            # add the progenitors to the list of halos
            x_progs = torch.cat([x_progs, t_next.repeat(1, num_progs, 1)], dim=-1)
            x_progs = x_progs.squeeze(0)  # remove the batch dim
            x_progs = x_progs * x_scale + x_loc  # unnormalize

            halo_feats = torch.cat([halo_feats, x_progs], dim=0)

            # create halo index for the progneitors
            prog_index = [next_halo_index + i for i in range(num_progs)]
            halo_index = halo_index + prog_index
            halo_remain_index = halo_remain_index + prog_index
            halo_t_index = halo_t_index + [halo_curr_t_index + 1] * num_progs

            # create edge index for the progenitors
            edge_index[0] = edge_index[0] + [halo_curr_index] * num_progs
            edge_index[1] = edge_index[1] + prog_index
            next_halo_index += num_progs

            n_max_iter -= 1
            if n_max_iter == 0:
                print('Max number of iterations reached')

    tree = Data(x=halo_feats, edge_index=torch.tensor(edge_index, dtype=torch.long))
    return tree


def generate_forest(
    root_features, times_out, mode='A', generator=None, classifier=None, regressor=None,
    norm_dict=None, classifier_norm_dict=None, regressor_norm_dict=None,
    n_max_iter=100, device='cpu', verbose=True):
    """ Generate multiple trees

    NOTE: the root_features should have the dimnesion of (batch_size, n_feat) instead
    of (n_feat,)
    """
    if len(root_features) != len(times_out):
        raise ValueError('root_features and times_out must have the same length')

    # check if the proper model is provided
    if mode == 'A':
        if classifier is None or regressor is None:
            raise ValueError('classifier and regressor must be provided for mode A')
        generator_fn = generate_tree_A
        generator_kwargs = {
            'classifier': classifier, 'regressor': regressor,
            'classifier_norm_dict': classifier_norm_dict,
            'regressor_norm_dict': regressor_norm_dict}
    elif mode == 'B':
        if generator is None:
            raise ValueError('model must be provided for mode B')
        generator_fn = generate_tree_B
        generator_kwargs = {'generator': generator, 'norm_dict': norm_dict}
    else:
        raise ValueError(f'Invalid mode {mode}')

    # generate trees for each root halo
    time_start = time.time()

    tree_list = []
    root_halo_range = range(len(root_features))

    # Use tqdm for progress tracking if verbose is True
    if verbose:
        loop = tqdm(
            root_halo_range, desc="Generating trees",
            miniters=len(root_features) // 100)
    else:
        loop = root_halo_range

    for i in loop:
        root_halo_feat = root_features[i]
        t_out = times_out[i]
        tree = generator_fn(
            root_halo_feat=root_halo_feat, t_out=t_out, n_max_iter=n_max_iter,
            device=device, **generator_kwargs)
        # always return a CPU tensor for memory efficiency
        tree_list.append(tree.cpu())

    time_end = time.time()
    if verbose:
        print(f'Time elapsed: {time_end - time_start:.2f} s')

    return tree_list
