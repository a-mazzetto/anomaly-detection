"""Fraph Classification Demo"""
# %%
import os
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# %% Define the model
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels=64)
print(model)

# %% Colab training
from gnn.metrics import CrossEntropyAccuracyFromLogits

torch.manual_seed(12345)
model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
accuracy = CrossEntropyAccuracyFromLogits()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     accuracy.reset()
     model.eval()
     with torch.no_grad():
        total_loss = 0
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            accuracy.update(out.detach(), data.y)
            total_loss += criterion(out.detach(), data.y).item()
        return correct / len(loader.dataset), accuracy.compute(), \
            total_loss / len(loader)  # Derive ratio of correct predictions.

colab_history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': []}
for epoch in range(1, 171):
    train()
    train_acc, my_train_acc, train_loss = test(train_loader)
    test_acc, my_test_acc, test_loss = test(test_loader)
    colab_history["loss"].append(train_loss)
    colab_history["val_loss"].append(test_loss)
    colab_history["accuracy"].append(train_acc)
    colab_history["val_accuracy"].append(test_acc)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, '
          f'Test Acc: {test_acc:.4f}, My Train Acc: {my_train_acc:.4f}, My Test Acc: {my_test_acc:.4f}')

# %% Train the model
from gnn.train_loops import graph_classification_train_loop, plot_training_results
from gnn.early_stopping import EarlyStopping
from gnn.metrics import CrossEntropyAccuracyFromLogits

torch.manual_seed(12345)
model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

history = graph_classification_train_loop(model=model, optimizer=optimizer, loss_fn=criterion, num_epochs=170,
                                          train_loader=train_loader, val_loader=test_loader,
                                          metric=CrossEntropyAccuracyFromLogits(),
                                          early_stopping=EarlyStopping(patience=200),
                                          best_model_path=None, print_freq=10)

fig = plot_training_results([colab_history, history], ["colab", "mine"])
os.makedirs("./plots", exist_ok=True)
fig.savefig("./plots/graph_classification_demo.pdf")

# %%
