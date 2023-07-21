"""Classification of three graph families"""
# %% Imports
import os
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.nn import Linear, Dropout
from torch_geometric.nn.models import GCN, GraphSAGE, GIN
from torch_geometric.nn import global_mean_pool

from gnn_data_generation.three_graph_families_dataset import ThreeGraphFamiliesDataset

# %% Parse parameters
try:
    parser = argparse.ArgumentParser(description="Classify three graph families",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-my", "--mydataset", type=bool, default=True, help="Should use custom dataset?")
    parser.add_argument("-feat", "--use_node_features", type=bool, default=True, help="Use node features in my dataset")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-p", "--patience", type=int, default=10, help="Patience")

    parser.add_argument("-hidden", "--gnn_hidden_channels", type=int, default=64, help="Hidden dimension")
    parser.add_argument("-layers", "--gnn_num_layers", type=int, default=3, help="Number of message passing layers")
    args = parser.parse_args()
    config = vars(args)
    MY_DATASET = config["mydataset"]
    USE_NODE_FEATURES = config["use_node_features"]
    NUM_EPOCHS = config["epochs"]
    PATIENCE = config["patience"]
    HIDDEN_CHANNELS = config["gnn_hidden_channels"]
    NUM_LAYERS = config["gnn_num_layers"]
    print(config)
except:
    print("Setting default values, argparser failed!")
    MY_DATASET = True
    USE_NODE_FEATURES = True
    NUM_EPOCHS = 10
    PATIENCE = 10
    HIDDEN_CHANNELS = 64
    NUM_LAYERS = 3

# %% Load the data

if MY_DATASET:
    dataset = ThreeGraphFamiliesDataset(use_features=USE_NODE_FEATURES)
    loader = DataLoader(dataset, batch_size=16)
    num_classes = len(np.unique([_data.y.numpy() for _data in dataset]))
    train_idx, test_idx = torch.utils.data.random_split(
        dataset, (0.7, 0.3), torch.Generator().manual_seed(0))

    train_dataset = [dataset[_idx] for _idx in train_idx.indices]
    test_dataset = [dataset[_idx] for _idx in test_idx.indices]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
else:
    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root='data/TUDataset', name='MUTAG')

    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    num_classes = dataset.num_classes

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %% Model

class gcn_classification(torch.nn.Module):
    def __init__(self, num_classes, conv="GCN", hidden_channels=HIDDEN_CHANNELS,
                 num_layers=NUM_LAYERS, activation="relu", dropout=0.1):
        super().__init__()

        if conv == "GCN":
            self.gnn = GCN(in_channels=-1, hidden_channels=hidden_channels,
                           num_layers=num_layers, out_channels=hidden_channels,
                           dropout=dropout, act=activation, norm=None)
        elif conv == "SAGE":
            self.gnn = GraphSAGE(in_channels=-1, hidden_channels=hidden_channels,
                                 num_layers=num_layers, out_channels=hidden_channels,
                                 dropout=dropout, act=activation, norm=None)
        elif conv == "GIN":
            self.gnn = GIN(in_channels=-1, hidden_channels=hidden_channels,
                           num_layers=num_layers, out_channels=hidden_channels,
                           dropout=dropout, act=activation, norm=None, train_eps=True)
        else:
            raise NotImplementedError("Unknown convolution type")
            
        self.dropout = Dropout()
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if data.batch is not None else torch.zeros((data.num_nodes,), dtype=torch.int64)

        x = self.gnn(x, edge_index)
        
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.lin(x)

        return x

gcn_classification(num_classes=num_classes, conv="GCN")(next(iter(test_loader)))

gcn_classification(num_classes=num_classes, conv="GCN")(train_dataset[0])

# %% Training
from torch.nn import CrossEntropyLoss
from gnn.train_loops import graph_classification_train_loop, plot_training_results
from gnn.early_stopping import EarlyStopping
from gnn.metrics import CrossEntropyAccuracyFromLogits

# %% Launch GCN training

model = gcn_classification(num_classes=num_classes, conv="GCN")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)

gcn_history = graph_classification_train_loop(
    model=model,
    optimizer=optimizer,
    loss_fn=CrossEntropyLoss(),
    num_epochs=NUM_EPOCHS,
    train_loader=train_loader,
    val_loader=test_loader,
    metric=CrossEntropyAccuracyFromLogits(),
    early_stopping=EarlyStopping(patience=PATIENCE),
    best_model_path="./data/gcn_model.pt",
    print_freq=10)

# %% Launch SAGE Training

model = gcn_classification(num_classes=num_classes, conv="SAGE")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)

sage_history = graph_classification_train_loop(
    model=model,
    optimizer=optimizer,
    loss_fn=CrossEntropyLoss(),
    num_epochs=NUM_EPOCHS,
    train_loader=train_loader,
    val_loader=test_loader,
    metric=CrossEntropyAccuracyFromLogits(),
    early_stopping=EarlyStopping(patience=PATIENCE),
    best_model_path="./data/sage_model.pt",
    print_freq=10)

# %% Launch GIN Training

model = gcn_classification(num_classes=num_classes, conv="GIN")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)

gin_history = graph_classification_train_loop(
    model=model,
    optimizer=optimizer,
    loss_fn=CrossEntropyLoss(),
    num_epochs=NUM_EPOCHS,
    train_loader=train_loader,
    val_loader=test_loader,
    metric=CrossEntropyAccuracyFromLogits(),
    early_stopping=EarlyStopping(patience=PATIENCE),
    best_model_path="./data/gin_model.pt",
    print_freq=10)

# %% Final Plotting
fig = plot_training_results([gcn_history, sage_history, gin_history], ["GCN", "SAGE", "GIN"])
os.makedirs("./plots", exist_ok=True)
fig.savefig("./plots/three_families_graph_classification.pdf")

# %%
