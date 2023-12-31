
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    # data configuration
    cfg.data_name = 'GUREFT05-Nanc1'
    cfg.data_root = '/mnt/ceph/users/tnguyen/florah/datasets/experiments/'

    # logging configuration
    cfg.workdir = './logging/'
    cfg.name = 'GUREFT05-Nanc1.transfGenerator'

    # training configuration
    # batching and shuffling
    cfg.train_frac = 0.8
    cfg.train_batch_size = 1024
    cfg.num_workers = 4

    # evaluation configuration
    cfg.eval_batch_size = 1024

    # model configuration
    cfg.input_size = 3  # number of input features, including time
    cfg.num_classes = 2  # maximum number of ancestors per halo
    cfg.sum_features = False  # sum over transformer outputs instead of taking the last one
    cfg.num_samples_per_graph = 4  # number of samples to generate per graph
    cfg.d_time = 1
    cfg.d_time_projection = 64
    cfg.d_feat_projection = 64
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
    cfg.rnn = config_dict.ConfigDict()
    cfg.rnn.name = 'gru'
    cfg.rnn.output_size = 64
    cfg.rnn.hidden_size = 64
    cfg.rnn.num_layers = 2
    cfg.rnn.activation = config_dict.ConfigDict()
    cfg.rnn.activation.name = 'relu'
    cfg.flows = config_dict.ConfigDict()
    cfg.flows.name = 'maf'
    cfg.flows.hidden_size = 64
    cfg.flows.num_blocks = 2
    cfg.flows.num_layers = 2
    cfg.flows.activation = config_dict.ConfigDict()
    cfg.flows.activation.name = 'tanh'
    cfg.classifier = config_dict.ConfigDict()
    cfg.classifier.name = 'mlp'
    cfg.classifier.hidden_sizes = [64, 64]
    cfg.classifier.activation = config_dict.ConfigDict()
    cfg.classifier.activation.name = 'relu'

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
    cfg.num_epochs = 1000
    cfg.num_steps = 1000000
    cfg.patience = 100
    cfg.monitor = 'val_loss'
    cfg.mode = 'min'
    cfg.grad_clip = 0.5
    cfg.save_top_k = 5
    cfg.accelerator = 'gpu'

    return cfg