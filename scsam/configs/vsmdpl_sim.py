

from ml_collections import config_dict

def get_config():

    config = config_dict.ConfigDict()

    # run configuration
    config.binary_path = "/mnt/home/tnguyen/sc-sam/sc-sam.gureft/gf"
    config.workdir = "/mnt/ceph/users/tnguyen/florah/sc-sam/florah-tree"
    config.name = "vsmdpl_sim"
    config.box_name = 'vsmdpl'
    config.run_scsam = True  # enable/disable running SC-SAM after preparing

    # data configuration
    config.data_root = "/mnt/ceph/users/tnguyen/florah/datasets/experiments/fixed-time-steps/"
    config.data_name = "VSMDPL-Nanc2/data.4.pkl"
    config.num_max_files = 1
    config.num_max_trees = 10000
    config.num_max_nodes = 5000
    config.num_trees_per_file = 10000
    config.is_dfs = False

    # SC-SAM configuration
    config.num_snapshots = 151
    config.min_snaps = [150, ]
    config.max_snaps = [150, ]
    config.use_main_branch = 0

    return config