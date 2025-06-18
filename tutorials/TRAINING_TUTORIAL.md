# FLORAH Tree Generator - Training Tutorial

This tutorial will guide you through training a FLORAH tree generation model from scratch. The FLORAH (Fast Learning of Astrophysical Reionization with Adaptive Hierarchies) model learns to generate merger trees for dark matter halos in cosmological simulations.

## Prerequisites

- Python 3.8+
- PyTorch with GPU support (recommended)
- Access to training data (halo merger trees)
- Basic understanding of command-line interfaces

## Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd florah-tree
```

2. **Install dependencies**:
```bash
pip install torch pytorch-lightning ml-collections absl-py numpy pyyaml tensorboard
pip install torch-geometric  # For graph neural network components
```

## Understanding the Project Structure

```
florah-tree/
├── train_atg.py           # Main training script
├── configs/               # Configuration files
│   └── gru-gru/          # GRU-based model configs
├── models/               # Model implementations
├── datasets.py           # Data loading utilities
└── datasets/             # Training data (create this directory)
```

## Step 1: Prepare Your Data

### Data Format
The model expects halo merger tree data in a specific format. Your training data should be stored as pickle files containing graph structures representing merger trees.

1. **Create data directory**:
```bash
mkdir -p datasets/processed/your-dataset-name
```

2. **Place your training data**:
   - Data files should be named `data.0.pkl`, `data.1.pkl`, etc.
   - Each file contains a list of PyTorch Geometric `Data` objects
   - Each `Data` object represents a merger tree with node features and edge connections

### Data Structure Example
```python
# Each merger tree is represented as:
Data(
    x=tensor,      # Node features (halo properties like mass, formation time)
    edge_index=tensor,  # Edge connections (parent-child relationships)
    edge_attr=tensor,   # Edge features (if any)
    y=tensor       # Target labels (optional)
)
```

## Step 2: Create a Configuration File

Copy an existing configuration and modify it for your needs:

```bash
cp configs/gru-gru/gureft05-nprog3-dt2_6-z15.py configs/gru-gru/my-training-config.py
```

### Key Configuration Parameters

Edit `configs/gru-gru/my-training-config.py`:

```python
def get_config():
    config = config_dict.ConfigDict()

    # === BASIC SETTINGS ===
    config.workdir = '/path/to/your/logging/directory'  # Where to save results
    config.name = 'my-model-name'                       # Experiment name
    config.overwrite = True                             # Overwrite existing results
    config.accelerator = 'gpu'                          # Use 'cpu' if no GPU

    # === DATA CONFIGURATION ===
    config.data = data = config_dict.ConfigDict()
    data.root = "/path/to/your/datasets/processed/"     # Path to your data
    data.name = "your-dataset-name"                     # Dataset folder name
    data.num_files = 4                                  # Number of data files to use
    data.train_frac = 0.8                               # 80% for training, 20% for validation

    # === MODEL ARCHITECTURE ===
    config.model = model = config_dict.ConfigDict()
    model.d_in = 2                                      # Input feature dimension
    model.num_classes = 3                               # Number of tree action classes

    # Encoder (processes input sequences)
    model.encoder = encoder = config_dict.ConfigDict()
    encoder.name = 'gru'                                # Use GRU cells
    encoder.d_model = 32                                # Hidden dimension
    encoder.num_layers = 2                              # Number of layers

    # Decoder (generates output sequences)
    model.decoder = decoder = config_dict.ConfigDict()
    decoder.name = 'gru'
    decoder.d_model = 32
    decoder.num_layers = 2

    # === TRAINING PARAMETERS ===
    config.training = training = config_dict.ConfigDict()
    training.max_epochs = 100                           # Maximum training epochs
    training.train_batch_size = 64                      # Training batch size
    training.eval_batch_size = 64                       # Validation batch size
    training.patience = 10                              # Early stopping patience

    # === OPTIMIZER SETTINGS ===
    config.optimizer = optimizer = config_dict.ConfigDict()
    optimizer.name = 'AdamW'
    optimizer.lr = 1e-3                                # Learning rate
    optimizer.weight_decay = 1e-4                       # Weight decay

    return config
```

### Important Parameters to Adjust

1. **Data paths**: Update `data.root` and `data.name` to point to your dataset
2. **Model size**: Adjust `d_model` and `num_layers` based on your data complexity and computational resources
3. **Training settings**: Modify `max_epochs`, `batch_size`, and `lr` based on your needs
4. **Hardware**: Set `accelerator = 'cpu'` if you don't have GPU access

## Step 3: Start Training

### Basic Training Command

```bash
python train_atg.py --config=configs/gru-gru/my-training-config.py
```

### Monitor Training Progress

The training script will:
1. Create a logging directory with your experiment name
2. Save model checkpoints automatically
3. Log training metrics to TensorBoard

**View training progress**:
```bash
tensorboard --logdir=/path/to/your/logging/directory/my-model-name
```

Open your browser to `http://localhost:6006` to see:
- Training and validation loss curves
- Learning rate schedule
- Model performance metrics

### Understanding the Output

During training, you'll see output like:
```
INFO - Starting training run my-model-name at /path/to/logging/
INFO - Preparing dataloader...
INFO - Creating model...
INFO - Training model...
INFO - Training for 100 epochs.
INFO - Training with batch size 64.
```

## Step 4: Monitor and Adjust Training

### Common Issues and Solutions

1. **Out of Memory Error**:
   - Reduce `train_batch_size` and `eval_batch_size`
   - Reduce model size (`d_model`, `num_layers`)

2. **Training Loss Not Decreasing**:
   - Check your data format and paths
   - Try increasing learning rate
   - Verify data preprocessing

3. **Overfitting** (validation loss increases while training loss decreases):
   - Reduce model complexity
   - Increase `weight_decay`
   - Add more training data

4. **Training Too Slow**:
   - Increase batch size (if memory allows)
   - Use GPU acceleration
   - Reduce model size for faster iterations

### Early Stopping

The model automatically stops training when validation loss stops improving. The patience parameter controls how many epochs to wait:
- `patience = 10`: Stop if no improvement for 10 epochs
- `patience = 50`: More patient, suitable for larger models

## Step 5: Evaluate Your Trained Model

### Check Training Results

After training completes, check your logging directory:
```
your-logging-directory/
└── my-model-name/
    ├── config.yaml                    # Saved configuration
    ├── lightning_logs/
    │   ├── checkpoints/              # Model checkpoints
    │   │   ├── best-epoch=X-step=Y-val_loss=Z.ckpt
    │   │   └── last-epoch=X-step=Y-val_loss=Z.ckpt
    │   └── events.out.tfevents.*     # TensorBoard logs
```

### Key Files

- **Best checkpoint**: Model with lowest validation loss
- **Last checkpoint**: Most recent model state
- **config.yaml**: Complete configuration used for training
- **TensorBoard logs**: Training metrics and curves

## Advanced Training Options

### Resume Training from Checkpoint

To continue training from a saved checkpoint:

```python
# In your config file:
config.checkpoint = 'best-epoch=45-step=12000-val_loss=0.5432.ckpt'
```

### Multi-GPU Training

For faster training with multiple GPUs:

```python
# In your config file:
config.accelerator = 'gpu'
# The trainer will automatically use all available GPUs
```

### Custom Learning Rate Schedule

```python
# In your config file:
config.scheduler = scheduler = config_dict.ConfigDict()
scheduler.name = 'WarmUpCosineAnnealingLR'
scheduler.warmup_steps = 1000      # Warm up learning rate
scheduler.decay_steps = 50000      # Total decay steps
scheduler.eta_min = 1e-6          # Minimum learning rate
```

## Next Steps

Once training is complete, you can:
1. Use the trained model for inference (see `INFERENCE_TUTORIAL.md`)
2. Experiment with different architectures by modifying the config
3. Train on larger datasets for better performance
4. Fine-tune hyperparameters for your specific use case

## Troubleshooting

### Common Error Messages

1. **"Dataset not found"**: Check `data.root` and `data.name` paths
2. **"CUDA out of memory"**: Reduce batch size or model size
3. **"No module named 'models'"**: Run from the repository root directory
4. **"Configuration file not found"**: Check the config file path

### Getting Help

- Check the TensorBoard logs for training progress
- Review the saved `config.yaml` to verify settings
- Examine the console output for detailed error messages
- Ensure all required dependencies are installed

## Performance Tips

1. **Start small**: Begin with a small model and dataset to verify everything works
2. **Use GPU**: Training is much faster with GPU acceleration
3. **Monitor memory**: Watch GPU/CPU memory usage during training
4. **Save frequently**: The model automatically saves checkpoints, but verify they're being created
5. **Experiment iteratively**: Make small changes and compare results

This completes the training tutorial! Your trained model will be ready for generating merger trees using the inference pipeline.
