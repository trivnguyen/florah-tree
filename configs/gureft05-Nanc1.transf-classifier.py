
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    # data configuration
    cfg.data_name = 'GUREFT05-Nanc1'
    cfg.data_root = '/mnt/ceph/users/tnguyen/florah/datasets/experiments/'

    # logging configuration
    cfg.workdir = './logging/'
    cfg.name = 'GUREFT05-Nanc1.transfClassifier'

    # model configuration
    cfg.input_size = 3
    cfg.num_classes = 2
    cfg.sum_features = False
    cfg.num_samples_per_graph = 1
    cfg.d_time = 1
    cfg.d_time_projection = 64
    cfg.featurizer = config_dict.ConfigDict()
    cfg.featurizer.name = 'transformer'
    cfg.featurizer.d_model = 64
    cfg.featurizer.nhead = 4
    cfg.featurizer.num_encoder_layers = 3
    cfg.featurizer.dim_feedforward = 128
    cfg.featurizer.batch_first = True
    cfg.featurizer.use_embedding = True
    cfg.featurizer.activation = config_dict.ConfigDict()
    cfg.featurizer.activation.name = 'Identity'
    cfg.classifier = config_dict.ConfigDict()
    cfg.classifier.name = 'mlp'
    cfg.classifier.hidden_sizes = [64, 64]
    cfg.classifier.activation = config_dict.ConfigDict()
    cfg.classifier.activation.name = 'ReLU'

    # training configuration
    # batching and shuffling
    cfg.train_frac = 0.8
    cfg.train_batch_size = 1024
    cfg.num_workers = 4

    # evaluation configuration
    cfg.eval_batch_size = 1024

    # optimizer and scheduler configuration
    cfg.optimizer = config_dict.ConfigDict()
    cfg.optimizer.name = 'AdamW'
    cfg.optimizer.lr = 5e-4
    cfg.optimizer.betas = (0.9, 0.98)
    cfg.optimizer.weight_decay = 1e-4
    cfg.optimizer.eps = 1e-9
    cfg.scheduler = config_dict.ConfigDict()
    cfg.scheduler.name = 'CosineAnnealingLR'
    cfg.scheduler.T_max = 100

    # training loop configuration
    cfg.num_epochs = 100
    cfg.num_steps = 100000
    cfg.patience = 50
    cfg.monitor = 'val_loss'
    cfg.mode = 'min'
    cfg.grad_clip = 0.5
    cfg.save_top_k = 5
    cfg.accelerator = 'gpu'

    return cfg