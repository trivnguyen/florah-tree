

from ml_collections import config_dict

def get_config():

    config = config_dict.ConfigDict()

    # run configuration
    config.binary_path = "/mnt/home/tnguyen/sc-sam/sc-sam.gureft/gf"
    config.workdir = "/mnt/ceph/users/tnguyen/florah/sc-sam/florah-tree"
    config.name = "gureft_gen"
    config.box_name = 'gureft'
    config.run_scsam = True  # enable/disable running SC-SAM after preparing

    # data configuration
    config.data_root = "/mnt/ceph/users/tnguyen/florah/generated_dataset/fixed-time-steps/"
    config.data_name = "GUREFT05-Nanc2-B"
    config.num_max_files = 100
    config.num_max_trees = 100_000
    config.num_max_nodes = 1000
    config.num_trees_per_file = 10000
    config.is_dfs = False

    # SC-SAM configuration
    config.num_snapshots = 171
    config.min_snaps = [170, ]
    config.max_snaps = [170, ]
    config.use_main_branch = 0

    return config