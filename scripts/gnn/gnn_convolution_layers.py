"""Small test of GNN convolution layers
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html
"""
# %% Imports
from typing import List, Optional
import numpy as np
import torch
from torch.nn import Linear, ReLU, Dropout, LayerNorm, LazyLinear, Sequential, Softplus
import torch.nn.functional as F
from torch_geometric.nn import Sequential as GeomSequential
from torch_geometric.nn.conv import GCNConv, SAGEConv, GINConv
from torch_geometric.nn.models import GCN, GraphSAGE, GIN
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# %% Small test dataset

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [3, 1]], dtype=torch.long)
edge_weight = torch.tensor([1, 2, 2, 4], dtype=torch.float)
x = torch.tensor([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0.5, 0.5, 1]], dtype=torch.float)

# Data takes the following parameters:
# :param x:          feature matrix (num_nodes, num_node_features)
# :param edge_index: (optional) graph connectivity with shape (2, num_edges)
# :param edge_attr:  (optional) edge feature matrix with dimension (num_edges, num_edge_features)
# :param y:          (optional) graph-level or nodelevel ground throught labels
# :param pos:        (optional) node position matrix with shape (num_nodes, num_dimensions)
data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_weight=edge_weight)

# Validate dataset
print(data.validate())

# Print data
print(data)

# Visualize batching index
print(data.batch)

# %% Multilayer perceptron

class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.):
        super(MLP, self).__init__()
        layers = []
        for in_dim, dim in zip([input_dim] + hidden_dims, hidden_dims):
            layers.extend([
                Linear(in_dim, dim),
                ReLU(),
                LayerNorm(dim),
                Dropout(dropout),
            ])
        self.layers = Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)
    
mlp = MLP(input_dim=4, hidden_dims=[10, 20, 10])
print(mlp)

# %% Test convolutional layers

gcn_conv = GCNConv(in_channels=-1, out_channels=5)
sage_conv = SAGEConv(in_channels=-1, out_channels=5)
gin_conv = GINConv(nn=MLP(input_dim=data.x.size(1), hidden_dims=[5, 5, 5]), train_eps=True)

print("GCN forward call:")
print(gcn_conv.forward(data.x, data.edge_index, data.edge_weight))

print("SAGE forward call:")
print(sage_conv.forward(data.x, data.edge_index))

print("GIN forward call:")
print(gin_conv.forward(data.x, data.edge_index))

print("Linear layer check:")
print(sage_conv.lin_r)

# %% Multiple layers are combined together to create the GNN model
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GIN.html

gcn = GCN(
    in_channels=-1,
    hidden_channels=10,
    num_layers=50,
    out_channels=None,
    dropout=0.1,
    act="relu",
    norm=None)

print("GCN model call:")
print(gcn(data.x, data.edge_index, edge_weight=edge_weight))

sage = GraphSAGE(
    in_channels=-1,
    hidden_channels=10,
    num_layers=50,
    out_channels=None,
    dropout=0.1,
    act="relu",
    norm=None)

print("SAGE model call:")
print(sage(data.x, data.edge_index))

# The NN cannot be passed from outside, but could extend the class
gin = GIN(
    in_channels=-1,
    hidden_channels=10,
    num_layers=50,
    out_channels=None,
    dropout=0.1,
    act="relu",
    norm=None,
    train_eps=True)

print("GIN model call:")
gin(data.x, data.edge_index)

# %% Test on dataset

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

print(dataset)

train_idx, test_idx = torch.utils.data.random_split(
    dataset, (0.7, 0.3), torch.Generator().manual_seed(0))

train_dataset = dataset[train_idx.indices].shuffle()
test_dataset = dataset[test_idx.indices]

print(dataset.y.unique())

print(dataset.num_classes)

print(dataset.num_node_features)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

for batch in train_loader:
    print(batch)

for batch in test_loader:
    print(batch)

print(data.batch)
# %% Classification model
from torch_geometric.nn import global_mean_pool

class gcn_classification(torch.nn.Module):
    def __init__(self, num_classes, conv="GCN"):
        super().__init__()

        hidden_channels = 32
        num_layers = 5
        activation = "relu"
        dropout = 0.1

        if conv == "GCN":
            self.gnn = GCN(
                in_channels=-1,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=hidden_channels,
                dropout=dropout,
                act=activation,
                norm=None)
        elif conv == "SAGE":
            self.gnn = GraphSAGE(
                in_channels=-1,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=hidden_channels,
                dropout=dropout,
                act=activation,
                norm=None)
        elif conv == "GIN":
            self.gnn = GIN(
                in_channels=-1,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=hidden_channels,
                dropout=dropout,
                act=activation,
                norm=None,
                train_eps=True)
        else:
            raise NotImplementedError("Unknown convolution type")
            
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if data.batch is not None else torch.zeros((data.num_nodes,), dtype=torch.int64)

        x = self.gnn(x, edge_index)
        
        x = global_mean_pool(x, batch)

        return self.lin(x)

print(gcn_classification(num_classes=dataset.num_classes, conv="GCN")(next(iter(test_loader))))

print(gcn_classification(num_classes=dataset.num_classes, conv="GCN")(train_dataset[0]))
# %% Training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = gcn_classification(num_classes=dataset.num_classes, conv="GIN").to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Create train loop
def train(model, optimizer, num_epochs, train_dl, val_dl):
    train_losses = []
    val_losses = []
    
    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Metrics
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(0, num_epochs):
        # initialize train and validation loss to 0
        train_loss = 0.0
        val_loss = 0.0
        
        train_acc = 0.0
        val_acc = 0.0

        model.train()

        for batch in train_dl:
            batch = batch.to(device)
            optimizer.zero_grad()
            predict = model(batch)
            loss = loss_fn(predict, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        
        train_correct = 0
        for batch in train_dl:
            batch = batch.to(device)
            class_predict = model(batch).argmax(dim=1)
            train_correct += int((class_predict == batch.y).sum())

        for batch in val_dl:
            batch = batch.to(device)
            predict = model(batch)
            loss = loss_fn(predict, batch.y)

            val_loss += loss.item()
            
        val_correct = 0
        for batch in val_dl:
            batch = batch.to(device)
            class_predict = model(batch).argmax(dim=1)
            val_correct += int((class_predict == batch.y).sum())

        train_loss = train_loss / len(train_dl) 
        val_loss = val_loss / len(val_dl)
        
        train_acc = train_correct / len(train_dl.dataset)
        val_acc = val_correct / len(val_dl.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if epoch % 10 == 0:
            print('Epoch: {} - Train Loss: {:.4f} - Validation Loss: {:.4f}'.format(epoch+1, train_loss, val_loss))   
    return {'loss': train_losses, 'val_loss': val_losses, 'acc': train_accuracies, 'val_acc': val_accuracies}

# %% Train

gin_history = train(model=model, optimizer=optimizer, num_epochs=50, train_dl=train_loader, val_dl=test_loader)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

fig, ax = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()

# ax[0, 0].plot(gcn_history['loss'], alpha=0.2, color="blue")
# ax[0, 0].plot(moving_average(gcn_history['loss'], 10), color="blue", label="GCN")
# ax[0, 0].plot(sage_history['loss'], alpha=0.2, color="red")
# ax[0, 0].plot(moving_average(sage_history['loss'], 10), color="red", label="SAGE")
ax[0, 0].plot(gin_history['loss'], alpha=0.2, color="blue")
ax[0, 0].plot(moving_average(gin_history['loss'], 10), color="blue", label="GIN")
ax[0, 0].set_title('Train loss')
ax[0, 0].grid()
ax[0, 0].legend()

# ax[0, 1].plot(gcn_history['acc'], alpha=0.2, color="blue")
# ax[0, 1].plot(moving_average(gcn_history['acc'], 10), color="blue", label="GCN")
# ax[0, 1].plot(sage_history['acc'], alpha=0.2, color="red")
# ax[0, 1].plot(moving_average(sage_history['acc'], 10), color="red", label="SAGE")
ax[0, 1].plot(gin_history['acc'], alpha=0.2, color="blue")
ax[0, 1].plot(moving_average(gin_history['acc'], 10), color="blue", label="GIN")
ax[0, 1].set_title('Train accuracy')
ax[0, 0].grid()
ax[0, 1].legend();

# ax[1, 0].plot(gcn_history['val_loss'], alpha=0.2, color="blue")
# ax[1, 0].plot(moving_average(gcn_history['val_loss'], 10), color="blue", label="GCN")
# ax[1, 0].plot(sage_history['val_loss'], alpha=0.2, color="red")
# ax[1, 0].plot(moving_average(sage_history['val_loss'], 10), color="red", label="SAGE")
ax[1, 0].plot(gin_history['val_loss'], alpha=0.2, color="blue")
ax[1, 0].plot(moving_average(gin_history['val_loss'], 10), color="blue", label="GIN")
ax[1, 0].set_title('Validation loss')
ax[1, 0].grid()
ax[1, 0].legend()

# ax[1, 1].plot(gcn_history['val_acc'], alpha=0.2, color="blue")
# ax[1, 1].plot(moving_average(gcn_history['val_acc'], 10), color="blue", label="GCN")
# ax[1, 1].plot(sage_history['val_acc'], alpha=0.2, color="red")
# ax[1, 1].plot(moving_average(sage_history['val_acc'], 10), color="red", label="SAGE")
ax[1, 1].plot(gin_history['val_acc'], alpha=0.2, color="blue")
ax[1, 1].plot(moving_average(gin_history['val_acc'], 10), color="blue", label="GIN")
ax[1, 1].set_title('Validation accuracy')
ax[1, 0].grid()
ax[1, 1].legend()

# %%
