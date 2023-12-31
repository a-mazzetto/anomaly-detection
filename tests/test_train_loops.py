"""Testing of training loop"""
import os
import numpy as np
import pytest
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from gnn.train_loops import graph_classification_train_loop, vae_training_loop
from gnn.metrics import CrossEntropyAccuracyFromLogits
from gnn.early_stopping import EarlyStopping

from gnn.vae import VAE
from gnn_data_generation.three_graph_families_dataset import ThreeGraphFamiliesDataset

from utils import create_results_folder, get_baseline_folder

@pytest.mark.parametrize("test_name, patience",
                         [
                             ("class_train_loop", 1000),
                             ("class_train_loop_with_pat", 10)
                         ]
)
def test_class_train_loop(test_name, patience):
    """Test classification training loop"""
    file_name = f'{test_name}.npy'
    baseline_file = os.path.join(get_baseline_folder(test_name), file_name)
    results_file = os.path.join(create_results_folder(test_name), file_name)

    # Dataset
    dataset = TUDataset(root="./data/MUTAG", name="MUTAG")

    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model definition
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
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
        
    # Initializations
    model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    _ = graph_classification_train_loop(
        model=model, optimizer=optimizer, loss_fn=criterion, num_epochs=170, train_loader=train_loader,
        val_loader=test_loader, metric=CrossEntropyAccuracyFromLogits(),
        early_stopping=EarlyStopping(patience=patience), best_model_path=None)
    
    # Evaluate model on one batch
    pred = model(next(iter(test_loader))).detach().numpy()
    np.save(results_file, pred)
    np.testing.assert_allclose(pred, np.load(baseline_file))

@pytest.mark.parametrize("test_name, n_prior_modes, use_graph_enc, pos_weight",
                         [
                             ("vae_train", 0, False, None),
                             ("vae_train_3modes", 0, False, None),
                             ("vae_train_graphenc", 0, True, None),
                             ("vae_train_posweight", 0, False, 1.),
                         ]
)
def test_vae_train_loop(test_name, n_prior_modes, use_graph_enc, pos_weight):
    """Test classification training loop"""
    file_name = f'{test_name}.npy'
    baseline_file = os.path.join(get_baseline_folder(test_name), file_name)
    results_file = os.path.join(create_results_folder(test_name), file_name)

    # Dataset
    dataset = ThreeGraphFamiliesDataset(use_features=True)

    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:300]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model definition
    device = torch.device("cpu")
    model = VAE(device=device, prior_modes=n_prior_modes, latent_dim=8, hidden_dim=16,
                n_gin_layers=3, graph_enc=use_graph_enc, pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, weight_decay=0.5e-4)

    # Training
    _ = vae_training_loop(model=model, optimizer=optimizer, num_epochs=10, train_dl=train_loader,
                          val_dl=test_loader, best_model_path=None, device=device,
                          early_stopping=EarlyStopping(patience=10))
    
    # Evaluate model on one batch
    model.eval()
    _, _, pred, _ = model(next(iter(test_loader)))
    pred = pred.detach().numpy()
    np.save(results_file, pred)
    np.testing.assert_allclose(pred, np.load(baseline_file))
