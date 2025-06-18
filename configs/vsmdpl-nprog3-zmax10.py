
from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # logging configuration
    config.workdir = '/mnt/ceph/users/tnguyen/florah-tree/logging/bug-fix/vsmdpl-nprog3-dt2_6-z10'
    config.name = 'gru-gru-3'
    config.overwrite = True
    config.enable_progress_bar = False
    config.checkpoint = None
    config.checkpoint_infer = 'best'
    config.accelerator = 'gpu'
    config.reset_optimizer = False

    # seed
    config.seed = seed = config_dict.ConfigDict()
    seed.data = 1531299
    seed.training = 13892179
    seed.inference = 82641312

    config.data = data = config_dict.ConfigDict()
    data.root = "/mnt/ceph/users/tnguyen/florah-tree/datasets/processed/"
    data.name = "vsmdpl-nprog3-dt2_6-z10"
    data.num_files = 15
    data.index_file_start = 0
    data.train_frac = 0.8
    data.reverse_time = False

    # inference dataset
    config.data_infer = data_infer = config_dict.ConfigDict()
    data_infer.root = "/mnt/ceph/users/tnguyen/florah-tree/datasets/generate_seeds"
    data_infer.name = "vsmdpl-test-nprog3-dt4-z10"
    data_infer.multiplicative_factor = 1
    data_infer.num_files = 5
    data_infer.index_file_start = 0
    data_infer.num_max_trees = 100_000_000
    data_infer.box = 'vsmdpl'
    data_infer.zmax = 10
    data_infer.batch_size = 4096
    data_infer.step = 4
    data_infer.job_id = 0
    data_infer.num_job = 1
    data_infer.outdir = f'/mnt/ceph/users/tnguyen/florah-tree/gen-datasets/testset/bug-fix/'\
        f'vsmdpl-nprog3-dt2_6-z10/{config.name}/z10-7'

   # model configuration
    config.model = model = config_dict.ConfigDict()
    model.d_in = 2
    model.num_classes= 3
    # history encoder args
    model.encoder = encoder = config_dict.ConfigDict()
    model.encoder.name = 'gru'
    model.encoder.d_model = 128
    model.encoder.d_out = 128
    model.encoder.dim_feedforward = 128
    model.encoder.num_layers = 4
    model.encoder.concat = False
    # progenitor encoder (decoder) args
    model.decoder = decoder = config_dict.ConfigDict()
    model.decoder.name = 'gru'
    model.decoder.d_model = 128
    model.decoder.d_out = 128
    model.decoder.dim_feedforward = 128
    model.decoder.num_layers = 4
    model.decoder.concat = False
    # npe args
    model.npe  = npe = config_dict.ConfigDict()
    model.npe.hidden_sizes = [128, 128]
    model.npe.num_transforms = 4
    model.npe.context_embedding_sizes = None
    model.npe.dropout = 0.2
    # classifier
    model.classifier = classifier = config_dict.ConfigDict()
    model.classifier.d_context = 1
    model.classifier.hidden_sizes = [128, 128]

    # optimizer and scheduler configuration
    config.optimizer = optimizer = config_dict.ConfigDict()
    optimizer.name = 'AdamW'
    optimizer.lr = 5e-5
    optimizer.betas = (0.9, 0.98)
    optimizer.weight_decay = 1e-4
    optimizer.eps = 1e-9
    config.scheduler = scheduler = config_dict.ConfigDict()
    scheduler.name = 'WarmUpCosineAnnealingLR'
    scheduler.decay_steps = 500_000  # include warmup steps
    scheduler.warmup_steps = 25_000
    scheduler.eta_min = 1e-6
    scheduler.interval = 'step'

    # training args
    config.training = training = config_dict.ConfigDict()
    training.max_epochs = 100_000
    training.max_steps = 500_000
    training.train_batch_size = 128
    training.eval_batch_size = 128
    training.monitor = 'val_loss'
    training.patience = 100_000
    training.save_top_k = 5
    training.save_last_k = 5
    training.gradient_clip_val = 0.5

    # model training args
    training.training_mode = 'all'
    training.use_sample_weight = False
    training.use_desc_mass_ratio = False
    training.num_branches_per_tree = 10
    training.freeze_args = freeze = config_dict.ConfigDict()
    freeze.encoder = False
    freeze.decoder = False
    freeze.npe = False
    freeze.classifier = False

    return config
