

from ml_collections import config_dict

def get_config():

    config = config_dict.ConfigDict()
    config.binary_path = "/mnt/home/tnguyen/sc-sam/sc-sam.gureft/gf"
    config.workdir = "/mnt/ceph/users/tnguyen/florah/sc-sam/florah-tree"
    config.name = "test_run"
    config.input_files = (
        "/mnt/ceph/users/tnguyen/florah/sc-sam/florah-tree/gureft05-nanc1-A/gen.dat",
        "/mnt/ceph/users/tnguyen/florah/sc-sam/florah-tree/gureft05-nanc1-A/gen.dat"
    )
    config.num_snapshots = 171
    config.min_snaps = [160, 170, ]
    config.max_snaps = [160, 170, ]
    config.use_main_branch = 0
    config.run_scsam = True

    return config