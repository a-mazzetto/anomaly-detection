"""Demo training of VGRNN with custom data"""
# %% Imports
import os
import argparse
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from vgrnn.vgrnn import VGRNN
from gnn_data_generation.vgrnn_test_dataset import VGRNNTestDataset
from gnn.early_stopping import EarlyStopping
from gnn.train_loops import vgrnn_train_loop, plot_vgrnn_training_result

# %% Initialization
try:
    parser = argparse.ArgumentParser(description="Classify three graph families",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-feat", "--use_node_features", action="store_false", help="Use node features in my dataset")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-p", "--patience", type=int, default=10, help="Patience")

    args = parser.parse_args()
    config = vars(args)

    USE_NODE_FEATURES = config["use_node_features"]
    NUM_EPOCHS = config["epochs"]
    PATIENCE = config["patience"]
    print(config)
except:
    print("Setting default values, argparser failed!")
    USE_NODE_FEATURES = True
    NUM_EPOCHS = 10
    PATIENCE = 10

# %% Load dataset
dataset = VGRNNTestDataset(use_features=USE_NODE_FEATURES)
loader = DataLoader(dataset, batch_size=32)
train_idx, test_idx = torch.utils.data.random_split(
    dataset, (0.7, 0.3), torch.Generator().manual_seed(0))

train_dataset = [dataset[_idx] for _idx in train_idx.indices]
test_dataset = [dataset[_idx] for _idx in test_idx.indices]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# %% Model

vgrnn_model = VGRNN(x_dim=dataset.x.size(-1), h_dim=32, z_dim=16, n_layers=1, eps=1e-10, bias=True)

optimizer = torch.optim.Adam(vgrnn_model.parameters(), lr=1e-2)

# %% Training

history = vgrnn_train_loop(model=vgrnn_model, optimizer=optimizer, num_epochs=NUM_EPOCHS,
                           train_loader=train_dataset, val_loader=test_dataset,
                           early_stopping=EarlyStopping(patience=PATIENCE),
                           best_model_path="./data/vgrnn_demo_model.pt", print_freq=10)

# %% Plotting

fig = plot_vgrnn_training_result(history=history)
os.makedirs("./plots", exist_ok=True)
fig.savefig("./plots/vgrnn_training.pdf")
