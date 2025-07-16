# FLORAH: A Generative Model for Dark Matter Halo Merger Trees

**FLORAH** is a generative model for creating realistic dark matter halo merger trees. It utilizes recurrent neural networks and normalizing flows to learn from N-body simulations and generate complete merger tree graph structures. FLORAH can accurately reproduces key merger rate statistics across a wide range of mass and redshift, outperforming conventional approaches based on the Extended Press-Schechter formalism.

This repository contains the code for FLORAH, as presented in Nguyen et al. (in prep).

## Overview

Merger trees are fundamental for understanding the hierarchical assembly of dark matter halos and are crucial inputs for semi-analytic models (SAMs) of galaxy formation.
FLORAH addresses the limitations of traditional merger tree construction methods by:
- Learning directly from cosmological N-body simulation data (e.g., Very Small MultiDark Planck).
- Representing merger trees as direct acyclic graph structures.

We show that FLORAH can generate merger trees that:
- Generate trees that accurately reproduce a wide range of statistical properties (mass functions, progenitor mass functions, etc.) across different redshifts.
- Demonstrate compatibility with SAMs, producing galaxy-halo scaling relations consistent with simulation-based SAM outputs.


## Quick Start

### Public Data and Pre-trained Models
Pre-trained models and example datasets are available on the Flatiron Institute's Rusty at `/mnt/ceph/users/tnguyen/public/florah-trees`. If you don't have access to this directory, you can download both the pre-trained model and example datasets from Dropbox:
- https://www.dropbox.com/scl/fo/mecsi8cfhgcp1bvdk36b5/ANJ6p6ZLqastyT9D8IzKGpY?rlkey=9wmvxppce86sm1hc3cfd5a3do&st=ecju2ah7&dl=0

### Prerequisites

- Python 3.8 or higher
- PyTorch (version compatible with PyTorch Lightning and PyTorch Geometric)
- PyTorch Lightning
- PyTorch Geometric
- ML Collections
- NumPy

If you're planning to train FLORAH on your simulation, I recommend using a GPU with CUDA support for training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone http://github.com/trivnguyen/florah-tree/
    cd florah-tree
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv florah_env
    source florah_env/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch pytorch-lightning torch-geometric ml-collections absl-py numpy tqdm
    ```
    *Note: Ensure your PyTorch installation matches your CUDA version. PyTorch Geometric is a bit finicky and might require specific PyTorch versions; consult its documentation if you encounter issues.*

4.  **Verify installation:**
    ```bash
    python -c "import florah_tree; print('FLORAH installed successfully!')"
    ```

## Usage

The primary scripts for interacting with FLORAH are `train_atg.py` for training new models and `infer_atg.py` for generating merger trees using pre-trained models.

### Configuration

FLORAH uses configuration files written with `ml_collections`. Examples can be found in the `configs/` directory. Key aspects of the configuration include:
-   **Data paths:** Specifying the location of training and inference datasets.
-   **Model architecture:** Defining parameters for the recurrent neural networks (e.g., GRU), normalizing flows, and classifier.
-   **Training parameters:** Learning rate, batch size, number of epochs, etc.
-   **Inference parameters:** Number of trees to generate, output directory, etc.

An example configuration file is `configs/final-models/vsmdpl-nprog3-zmax10.py`. The config file can be a bit complex, so please feel free to reach out if you have questions about specific parameters.

### Training a New Model

To train FLORAH on your own merger tree dataset:

1.  **Prepare your dataset:**
    Merger trees should be processed into a format readable by the `datasets.py` script. This typically involves converting simulation outputs into collections of PyTorch Geometric `Data` objects, where each object represents a tree. Each node in the tree should have features (e.g., halo mass, redshift), and edges should represent progenitor-descendant relationships.

2.  **Create or modify a configuration file:**
    Adjust an existing configuration file (e.g., `configs/final-models/vsmdpl-nprog3-zmax10.py`) to point to your dataset and set desired model and training parameters.

3.  **Run the training script:**
    ```bash
    python train_atg.py --config=path/to/your/config.py --workdir=path/to/your/logging_directory
    ```
    -   `--config`: Path to your configuration file.
    -   `--workdir`: (Optional) Directory where logs and checkpoints will be saved. Defaults to what's specified in the config file.

    Training progress, logs, and model checkpoints will be saved in the specified working directory.

### Generating Merger Trees (Inference)

To generate merger trees using a pre-trained FLORAH model:

1.  **Ensure you have a trained model checkpoint (`.ckpt` file).**
    Pre-trained models can be found at `/mnt/ceph/users/tnguyen/florah-tree-models` on Rusty or downloaded from Dropbox (see above).

2.  **Create or modify an inference configuration file:**
    This file will specify the path to the model checkpoint, parameters for the initial halo seeds (e.g., mass and redshift ranges), the number of trees to generate, and the output directory.

3.  **Run the inference script:**
    ```bash
    python infer_atg.py --config=path/to/your/inference_config.py
    ```
    Generated merger trees will be saved in the output directory specified in the configuration, typically as pickled lists of tree data structures.

## Project Structure

```
florah-tree/
├── florah_tree/          # Core FLORAH package
│   ├── __init__.py
│   ├── atg.py            # Main AutoregressiveTreeGen model class
│   ├── models.py         # Neural network components (MLP, GRU, etc.)
│   ├── flows.py          # Normalizing flow implementations
│   ├── transformer.py    # Transformer model components (if used)
│   ├── training_utils.py # Utility functions for training
│   ├── infer_utils.py    # Utility functions for inference
│   ├── analysis_utils.py # Utility functions for analyzing trees
│   └── utils.py          # General utility functions
├── configs/              # Configuration files for training and inference
├── datasets.py           # Script for reading and preparing datasets
├── train_atg.py          # Main script for training FLORAH models
├── infer_atg.py          # Main script for generating trees with FLORAH
├── tutorials/            # Jupyter notebooks for tutorials (if available)
├── README.md             # This file
└── LICENSE               # Project license
```

## Key Scripts and Their Roles

-   **`train_atg.py`**:
    -   Handles the entire training pipeline.
    -   Loads the dataset using `datasets.py`.
    -   Initializes the `AutoregTreeGen` model from `florah_tree/atg.py`.
    -   Uses PyTorch Lightning for the training loop, logging, and checkpointing.
    -   Saves model checkpoints and training logs.

-   **`infer_atg.py`**:
    -   Loads a pre-trained `AutoregTreeGen` model from a checkpoint.
    -   Generates initial "seed" halos based on specified criteria (e.g., mass and redshift ranges from a dataset or custom inputs).
    -   Uses the model to autoregressively generate the full merger trees for these seed halos.
    -   Saves the generated trees to disk.

-   **`datasets.py`**:
    -   Contains functions to read raw merger tree data from simulations (e.g., consistent-trees output).
    -   Processes this data into PyTorch Geometric `Data` objects suitable for training.
    -   Handles normalization of features and creation of data loaders.

-   **`florah_tree/atg.py`**:
    -   Defines the `AutoregTreeGen` class, which is the core PyTorch Lightning module for FLORAH.
    -   Integrates the encoder (RNN/GRU), decoder, classifier (for number of progenitors), and normalizing flows (for continuous parameter generation).
    -   Implements the training, validation, and prediction steps.

-   **`florah_tree/models.py`**:
    -   Contains definitions for various neural network building blocks, such as Multi-Layer Perceptrons (MLPs) and Gated Recurrent Units (GRUs).

-   **`florah_tree/flows.py`**:
    -   Implements the normalizing flow models used for sampling progenitor properties.

## 📊 Model Architecture Highlights

FLORAH generates merger trees autoregressively:
1.  Given a halo, an **encoder** (typically a GRU) processes its properties and assembly history.
2.  A **classifier** predicts the number of progenitors this halo will have at the next time step.
3.  For each progenitor:
    a.  A **decoder** (another GRU) and **normalizing flows** are used to sample its properties (e.g., mass).
4.  This process is repeated recursively for each progenitor until the earliest time of interest is reached, forming a complete tree.

This graph-based generation allows FLORAH to capture the full branching structure of merger trees.

## Citation

Further publication details to be added here once available.

<!-- If you use FLORAH in your research, please cite the following paper: -->
<!-- Nguyen et al. (in prep). -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue on the GitHub repository or contact the corresponding author:
- Tri Nguyen (trivtnguyen@northwestern.edu)

---
