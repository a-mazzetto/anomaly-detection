"""Evaluate a graph given the model"""
# %% Imports
import torch
from gnn_data_generation.three_graph_families_dataset import ThreeGraphFamiliesDataset
from graph_vae import VAE

# %% Load model
state_dict = torch.load(
    r"C:\Users\user\git\anomaly-detection\data\trained_on_gpu\vae_demo_simple.pt",
    map_location=torch.device('cpu'))

# %%
model = VAE(device=torch.device("cpu"), latent_dim=32, hidden_dim=64)
model.load_state_dict(state_dict)
model.eval()
# %% Dataset
dataset = ThreeGraphFamiliesDataset(use_features=True)

# %% Predict
first_class = dataset[dataset.y == 0]
_, _, y_pred, y_true = model(first_class[0])
# %%
# Use Poisson Binomial Distribution
import torch.distributions as dist
mega_sample = model.decoder(dist.Normal(*model.encoder(first_class[0])).sample((1000,)))
p_pos = (mega_sample > 0.5).sum(dim=0) / mega_sample.size(0)
# %%
