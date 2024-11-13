

from ml_collections import config_dict

def get_config():

    config = config_dict.ConfigDict()

    # run configuration
    config.binary_path = "/mnt/home/tnguyen/sc-sam/sc-sam.gureft/gf"
    config.workdir = "/mnt/ceph/users/tnguyen/florah-tree/sc-sam/gureft90-nprog2-dt4-z15"
    config.name = ""
    config.box_name = 'gureft90'
    config.run_scsam = True  # enable/disable running SC-SAM after preparing

    # data configuration
    config.data_root = "/mnt/ceph/users/tnguyen/florah-tree/generated-datasets/testset/nprog2/gureft90-nprog2-dt2_6-z15"
    config.data_name = "gru-gru-opt-nopos/halos.0.pkl"
    config.start_file = 0
    config.num_max_files = 1
    config.num_max_trees = 10000
    config.num_max_nodes = 5000
    config.num_trees_per_file = 10000
    config.is_dfs = False

    # SC-SAM configuration
    config.num_snapshots = 171
    config.min_snaps = [170, ]
    config.max_snaps = [170, ]
    config.use_main_branch = 0

    return config