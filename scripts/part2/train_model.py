"""Demo training of VGRNN with custom data"""
# %% Imports
import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from gnn.dynvae import DynVAE
from gnn.early_stopping import EarlyStopping
from gnn.train_loops import vgrnn_train_loop, plot_vgrnn_training_result
from cuda.cuda_utils import select_device
from data_generation.kpath_dyngraphs import kpath_dyngraphs

# %% Initialization
try:
    parser = argparse.ArgumentParser(description="Classify three graph families",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-nbatch", "--batch_size", type=int, default=32, help="Batch size")

    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-p", "--patience", type=int, default=10, help="Patience")
    parser.add_argument("-pf", "--print_freq", type=int, default=1, help="Patience")

    parser.add_argument("-nmodes", "--n_prior_modes", type=int, default=2, help="Prior mixture dimension")

    parser.add_argument("-hdim", "--hidden_dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument("-ldim", "--latent_dim", type=int, default=16, help="Latent dimension")

    parser.add_argument("-ngin", "--n_gin_layers", type=int, default=1, help="Number of encoder GIN layers")
    parser.add_argument("-ndense", "--n_dense_layers", type=int, default=1, help="Number of prior encoder dense layers")
    parser.add_argument("-ngru", "--n_gru_layers", type=int, default=1, help="Number of GRU layers")

    parser.add_argument("-optstep", "--optimizer_step", type=float, default=5e-4, help="Optimizer step")
    parser.add_argument("-optdecay", "--optimizer_decay", type=float, default=5e-4, help="Optimizer step")

    args = parser.parse_args()
    config = vars(args)

    BATCH_DIM = config["batch_size"]

    NUM_EPOCHS = config["epochs"]
    PATIENCE = config["patience"]
    PRINT_FREQ = config["print_freq"]

    N_PRIOR_MODES = config["n_prior_modes"]

    HIDDEN_DIM = config["hidden_dim"]
    LATEN_DIM = config["latent_dim"]

    N_GIN_LAYERS = config["n_gin_layers"]
    N_DENSE_LAYERS = config["n_dense_layers"]
    N_GRU_LAYERS = config["n_gru_layers"]

    OPTIMIZER_STEP = config["optimizer_step"]
    OPTIMIZER_DECAY = config["optimizer_decay"]
    print(config)
except:
    print("Setting default values, argparser failed!")
    BATCH_DIM = 32

    NUM_EPOCHS = 2
    PATIENCE = 2
    PRINT_FREQ = 1

    N_PRIOR_MODES = 0

    HIDDEN_DIM = 16
    LATEN_DIM = 32

    N_GIN_LAYERS = 3
    N_DENSE_LAYERS = 3
    N_GRU_LAYERS = 5

    OPTIMIZER_STEP = 5e-4
    OPTIMIZER_DECAY = 5e-4

# %% Dataset
dataset = kpath_dyngraphs(
    dataset_file="./data/dataset_003/auth_ddcrp.txt",
    scores_file="./data/dataset_003/DP/phase_4_source_score.txt",
    max_t=24 * 60 * 60,
    interval=15 * 60,
    n_intervals=4,
    k_path_len=3)

train_idx, test_idx = torch.utils.data.random_split(
    dataset, (0.7, 0.3), torch.Generator().manual_seed(0))

train_dataset = [dataset[_idx] for _idx in train_idx.indices]
test_dataset = [dataset[_idx] for _idx in test_idx.indices]

train_loader = DataLoader(train_dataset, batch_size=BATCH_DIM, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_DIM)

# %% Model

device = select_device()

dynvae_model = DynVAE(x_dim=dataset.x.size(-1), h_dim=HIDDEN_DIM, z_dim=LATEN_DIM, eps=1e-10, n_prior_modes=N_PRIOR_MODES,
                      n_gin_layers=N_GIN_LAYERS, n_dense_layers=N_DENSE_LAYERS, n_gru_layers=N_GRU_LAYERS, bias=True,
                      device=device)

optimizer = torch.optim.Adam(dynvae_model.parameters(), lr=OPTIMIZER_STEP, weight_decay=OPTIMIZER_DECAY)

# %% Training

history = vgrnn_train_loop(model=dynvae_model, optimizer=optimizer, num_epochs=NUM_EPOCHS,
                           train_loader=train_dataset, val_loader=test_dataset,
                           early_stopping=EarlyStopping(patience=PATIENCE),
                           best_model_path="./data/dynvae_part2_model.pt", print_freq=PRINT_FREQ,
                           device=device)

# %% Plotting

fig = plot_vgrnn_training_result(history=history)
os.makedirs("./plots", exist_ok=True)
fig.savefig("./plots/dynvae_part2_training.pdf")

# %%
