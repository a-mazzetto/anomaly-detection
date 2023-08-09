"""Graph Variational Autoencoder"""

# %% Imports
import os
import argparse
import torch
import torch.distributions as dist
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GIN, global_mean_pool
from torch_geometric.utils import to_dense_adj
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

    parser.add_argument("-bgenc", "--b_graph_enc", action="store_false", help="Use graph encoder?")
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

# %% Prior distribution
latent_dim = 8

def get_prior(device, num_modes, latent_dim):
    """
    This function should create an instance of a MixtureSameFamily distribution
    according to the above specification.
    The function takes the num_modes and latent_dim as arguments, which should
    be used to define the distribution.
    Your function should then return the distribution instance.
    """
    probs = torch.ones(num_modes, device=device) / num_modes
    categorical = dist.Categorical(probs=probs)
    loc = torch.randn((num_modes, latent_dim), device=device)
    scale = torch.ones((num_modes, latent_dim), device=device)
    scale = torch.nn.functional.softplus(scale)
    prior = dist.MixtureSameFamily(
        mixture_distribution=categorical,
        component_distribution=dist.Independent(dist.Normal(loc=loc, scale=scale), 1))
    return prior

# Simple Gaussian
# prior = dist.Normal(loc=torch.zeros(latent_dim), scale=1.)
# Gaussian Mixture
# prior = get_prior(num_modes=N_PRIOR_MODES, latent_dim=LATENT_DIM, device=torch.device("cpu"))

# %% Encoder
class Encoder(torch.nn.Module):
    def __init__(self, latent_dim=8, hidden_dim=16, n_gin_layers=10):
        super().__init__()
        self.gin = GIN(in_channels=-1, hidden_channels=hidden_dim, num_layers=n_gin_layers, dropout=0.2)
        self.mean = torch.nn.Linear(hidden_dim, latent_dim)
        self.std = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.Softplus()
        )

    def forward(self, data, addition=None):
        hidden = self.gin(data.x, data.edge_index)
        if addition is not None:
            hidden += addition
        return self.mean(hidden), self.std(hidden)

encoder = Encoder()
loc, scale = encoder(dataset[0])

# %% Graph Encoder
class GraphEncoder(torch.nn.Module):
    """Lern a full Graph embedding and replicate the vector for each node"""
    def __init__(self, hidden_dim=16, n_gin_layers=10):
        super().__init__()
        self.gin = GIN(in_channels=-1, hidden_channels=hidden_dim, num_layers=n_gin_layers, dropout=0.2)

    def forward(self, data):
        hidden = self.gin(data.x, data.edge_index)
        hidden_pool = global_mean_pool(hidden, data.batch)
        hidden = hidden_pool[data.batch]
        return hidden

graph_encoder = GraphEncoder()
encoding = graph_encoder(dataset[0])

# %% Decoder

class InnerProductDecoder(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.dropout = torch.nn.Dropout(p=0.2)
    
    def forward(self, inp):
        inp = self.dropout(inp)
        x = torch.transpose(inp, dim0=-2, dim1=-1)
        x = torch.matmul(inp, x)
        if self.logits:
            return x
        else:
            return torch.sigmoid(x)

# decoder = InnerProductDecoder()
# decoder(prior.sample(torch.Size([2])))

class InnerProductDecoder2(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.dropout = torch.nn.Dropout(p=0.2)
    
    def forward(self, ld_in, ud_in):
        ld_in = self.dropout(ld_in)
        ud_in = self.dropout(ud_in)
        ud_in_t = torch.transpose(ud_in, dim0=-2, dim1=-1)
        x = torch.matmul(ld_in, ud_in_t)
        if self.logits:
            return x
        else:
            return torch.sigmoid(x)

# %% VAE
class VAE(torch.nn.Module):
    def __init__(self, device, latent_dim=8, hidden_dim=16, n_gin_layers=10, prior_modes=0,
                 graph_enc=False, pos_weight=None):
        super().__init__()

        # prior
        self.simple_prior = prior_modes < 1
        self.prior = dist.Normal(loc=torch.zeros(latent_dim).to(device), scale=1.) if \
            self.simple_prior else get_prior(device, num_modes=prior_modes, latent_dim=LATENT_DIM)

        # encoder, decoder
        self.encoder_ld = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim, n_gin_layers=n_gin_layers)
        self.encoder_ud = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim, n_gin_layers=n_gin_layers)
        self.b_graph_enc = graph_enc
        if graph_enc:
            self.graph_encoder = GraphEncoder(hidden_dim=hidden_dim, n_gin_layers=n_gin_layers)

        # Work with logits
        self.decoder = InnerProductDecoder2(logits=True)

        # Loss
        self.pos_weight = pos_weight

    def forward(self, data):
        if self.b_graph_enc:
            graph_enc = self.graph_encoder(data)
        else:
            graph_enc = None
        mu_ld, std_ld = self.encoder_ld(data, addition=graph_enc)
        mu_ud, std_ud = self.encoder_ud(data, addition=graph_enc)
        z_ld = self.sampler(mu_ld, std_ld)
        z_ud = self.sampler(mu_ud, std_ud)
        if data.batch is None:
            batch = torch.zeros(data.x.size(0), dtype=torch.int)
        else:
            batch = data.batch
        n_nodes = torch.bincount(batch).tolist()
        max_nodes = max(n_nodes)
        z_ld = torch.split(z_ld, n_nodes, dim=0)
        z_ud = torch.split(z_ud, n_nodes, dim=0)
        adj_pred = torch.stack(
            [torch.nn.functional.pad(
                self.decoder(_z_ld, _z_ud),
                (0, max(0, max_nodes - _z_ld.size(0)), 0, max(0, max_nodes - _z_ld.size(0))),
                mode="constant",
                value=0) for _z_ld, _z_ud in zip(z_ld, z_ud)], dim=0)

        # compute losses
        dense_adj = to_dense_adj(data.edge_index, data.batch)
        assert adj_pred.shape == dense_adj.shape
        assert not torch.any(torch.isnan(adj_pred))
        assert not torch.any(torch.isnan(dense_adj))
        nll_loss = self._bernoulli_loss(adj_pred, dense_adj)
        if self.simple_prior:
            kl_loss = self._gaussian_kl(mu_ld, std_ld) + self._gaussian_kl(mu_ud, std_ud)
        else:
            kl_loss = self._gaussian_mixture_kl(mu_ld, std_ld) + self._gaussian_mixture_kl(mu_ud, std_ud)
        return nll_loss, kl_loss, torch.sigmoid(adj_pred), dense_adj
    
    def sampler(self, mean, std):
        epsilon = self.prior.sample((mean.size(0),))
        z = mean + std * epsilon
        return z

    def _bernoulli_loss(self, y_pred, y_true):
        temp_size = torch.tensor(y_true.shape).prod()
        temp_sum = y_true.sum()
        if self.pos_weight is None:
            posw = float(temp_size - temp_sum) / temp_sum
        else:
            posw = self.pos_weight
        nll_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=y_pred, target=y_true, pos_weight=posw, reduction='sum')
        return nll_loss
    
    def _gaussian_kl(self, mean, scale):
        q = dist.Normal(mean, scale)
        # Sum across all
        return dist.kl_divergence(q, self.prior).sum()
    
    def _gaussian_mixture_kl(self, loc, scale):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Note: other people experience overflow here
        # See: https://discuss.pytorch.org/t/kld-loss-goes-nan-during-vae-training/42305/4
        eps = 1e-8
        std_log = torch.log(scale + eps)
        kld_element =  torch.sum(1 + 2 * std_log - loc.pow(2) - torch.pow(torch.exp(std_log), 2))
        return -0.5 * kld_element

# vae = VAE(prior_modes=N_PRIOR_MODES, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, device=torch.device("cpu"))
# vae_call = vae(dataset[0])
# vae_call_batch = vae(next(iter(train_loader)))

# %% Kick training
if __name__ == "__main__":
    device = select_device()

    model = VAE(device=device, prior_modes=N_PRIOR_MODES, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                n_gin_layers=N_GIN_LAYERS, graph_enc=USE_GRAPH_ENC, pos_weight=POS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=OPT_STEP, weight_decay=OPT_DECAY)

    history = vae_training_loop(model=model, optimizer=optimizer, num_epochs=NUM_EPOCHS,
        train_dl=train_loader, val_dl=test_loader, early_stopping=EarlyStopping(patience=PATIENCE),
        best_model_path="./data/vae_demo.pt", print_freq=PRINT_FREQ, device=device)
# %% Validate

# vae.eval()
# example = vae(dataset[10])
# %% Plotting
if __name__ == "__main__":
    fig = plot_vae_training_results(history=history)
    os.makedirs("./plots", exist_ok=True)
    fig.savefig("./plots/vae_training.pdf")

# %%
