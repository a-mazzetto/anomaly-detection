"""Graph Variational Autoencoder"""

# %% Imports
import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from gnn.vae import VAE
from gnn_data_generation.three_graph_families_dataset import ThreeGraphFamiliesDataset
from gnn_data_generation.one_graph_family_dataset import OneGraphFamilyDataset

from gnn.train_loops import vae_training_loop, plot_vae_training_results
from gnn.early_stopping import EarlyStopping
from cuda.cuda_utils import select_device

# %% Initialization
try:
    parser = argparse.ArgumentParser(description="Graph VAE example",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-gdat", "--giant_dataset", action="store_false", help="Use giant dataset?")
    parser.add_argument("-feat", "--use_node_features", action="store_false", help="Use node features in my dataset")
    parser.add_argument("-nbatch", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-optstep", "--optimizer_step", type=float, default=5e-4, help="Optimizer step")
    parser.add_argument("-optdecay", "--optimizer_decay", type=float, default=5e-4, help="Optimizer step")

    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-p", "--patience", type=int, default=10, help="Patience")
    parser.add_argument("-pf", "--print_freq", type=int, default=1, help="Patience")

    parser.add_argument("-lat", "--latent_dim", type=int, default=16, help="Latent dimension")
    parser.add_argument("-hid", "--hidden_dim", type=int, default=54, help="Hidden dimension")
    parser.add_argument("-ngl", "--n_gin_layers", type=int, default=10, help="Number GIN layers")
    parser.add_argument("-modes", "--n_modes", type=int, default=0, help="Mixture of Gaussians?")

    parser.add_argument("-bgenc", "--b_graph_enc", action="store_true", help="Use graph encoder?")
    parser.add_argument("-pwgt", "--pos_weight", type=float, default=None, help="Positive label weight")

    args = parser.parse_args()
    config = vars(args)

    USE_GIANT_DATASET = config["giant_dataset"]
    USE_NODE_FEATURES = config["use_node_features"]
    N_BATCH = config["batch_size"]
    OPT_STEP = config["optimizer_step"]
    OPT_DECAY = config["optimizer_decay"]

    NUM_EPOCHS = config["epochs"]
    PATIENCE = config["patience"]
    PRINT_FREQ = config["print_freq"]

    LATENT_DIM = config["latent_dim"]
    HIDDEN_DIM = config["hidden_dim"]
    N_GIN_LAYERS = config["n_gin_layers"]
    N_PRIOR_MODES = config["n_modes"]

    USE_GRAPH_ENC = config["b_graph_enc"]
    POS_WEIGHT = config["pos_weight"]

    print(config)
except:
    print("Setting default values, argparser failed!")

    USE_GIANT_DATASET = False
    USE_NODE_FEATURES = True
    N_BATCH = 32
    OPT_STEP = 1e-3
    OPT_DECAY = 1e-5

    NUM_EPOCHS = 20
    PATIENCE = 20
    PRINT_FREQ = 1

    LATENT_DIM = 16
    HIDDEN_DIM = 54
    N_GIN_LAYERS = 10
    N_PRIOR_MODES = 3

    USE_GRAPH_ENC = True
    POS_WEIGHT = None

SIMPLE_PRIOR = N_PRIOR_MODES < 1

# %% Dataset
if USE_GIANT_DATASET:
    dataset = OneGraphFamilyDataset(use_features=USE_NODE_FEATURES)
else:
    dataset = ThreeGraphFamiliesDataset(use_features=USE_NODE_FEATURES)
    # Use only one class
    dataset = dataset[dataset.y == 0]

train_idx, test_idx = torch.utils.data.random_split(
    dataset, (0.7, 0.3), torch.Generator().manual_seed(0))

train_dataset = [dataset[_idx] for _idx in train_idx.indices]
test_dataset = [dataset[_idx] for _idx in test_idx.indices]

train_loader = DataLoader(train_dataset, batch_size=N_BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=N_BATCH)

# %% Kick training, save best model and history plot
device = select_device()

model = VAE(device=device, prior_modes=N_PRIOR_MODES, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
            n_gin_layers=N_GIN_LAYERS, graph_enc=USE_GRAPH_ENC, pos_weight=POS_WEIGHT)
optimizer = torch.optim.Adam(model.parameters(), lr=OPT_STEP, weight_decay=OPT_DECAY)

history = vae_training_loop(model=model, optimizer=optimizer, num_epochs=NUM_EPOCHS,
    train_dl=train_loader, val_dl=test_loader, early_stopping=EarlyStopping(patience=PATIENCE),
    best_model_path="./data/vae_demo.pt", print_freq=PRINT_FREQ, device=device)

fig = plot_vae_training_results(history=history)
os.makedirs("./plots", exist_ok=True)
fig.savefig("./plots/vae_training.pdf")

# %%
