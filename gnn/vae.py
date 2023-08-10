"""Graph Variational Auto-encoder"""
# %% Imports
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.special import expit
import torch
import torch.distributions as dist
from torch_geometric.nn import GIN, global_mean_pool
from torch_geometric.utils import to_dense_adj

# %% Prior distribution
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

if __name__ == "__main__":
    latent_dim = 8
    prior = get_prior(torch.device("cpu"), 3, latent_dim)
    # Simple Gaussian
    prior = dist.Normal(loc=torch.zeros(latent_dim), scale=1.)
    # Gaussian Mixture
    prior = get_prior(num_modes=3, latent_dim=latent_dim, device=torch.device("cpu"))

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

if __name__ == "__main__":
    from gnn_data_generation.three_graph_families_dataset import ThreeGraphFamiliesDataset
    dataset = ThreeGraphFamiliesDataset(use_features=True)
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
        if data.batch is not None:
            hidden = hidden_pool[data.batch]
        else:
            hidden = hidden_pool
        return hidden

if __name__ == "__main__":
    graph_encoder = GraphEncoder()
    encoding = graph_encoder(dataset[0])

# %% Decoder

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

if __name__ == "__main__":
    decoder = InnerProductDecoder2(logits=True)
    decoder(prior.sample(torch.Size([2])),
            prior.sample(torch.Size([2])))
    
# %% VAE
class VAE(torch.nn.Module):
    def __init__(self, device, latent_dim=8, hidden_dim=16, n_gin_layers=10, prior_modes=0,
                 graph_enc=False, pos_weight=None):
        super().__init__()

        # prior
        self.simple_prior = prior_modes < 1
        self.prior = dist.Normal(loc=torch.zeros(latent_dim).to(device), scale=1.) if \
            self.simple_prior else get_prior(device, num_modes=prior_modes, latent_dim=latent_dim)

        # encoder, decoder
        self.encoder_ld = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim, n_gin_layers=n_gin_layers)
        self.encoder_ud = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim, n_gin_layers=n_gin_layers)
        self.b_graph_enc = graph_enc
        if graph_enc:
            self.graph_encoder = GraphEncoder(hidden_dim=hidden_dim, n_gin_layers=n_gin_layers)

        # Work with logits
        self.decoder = InnerProductDecoder2(logits=True)

        # Loss
        self.pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None

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

if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=32)
    model = VAE(device=torch.device("cpu"), prior_modes=0, latent_dim=2, hidden_dim=12,
                n_gin_layers=3, graph_enc=True, pos_weight=1.)
    model(dataset[0])
    model(next(iter(loader)))

# %% Model evaluation
def vae_score_given_model(model, datum, plots=True):
    """Assuming model of type Graph VAE"""
    # Bernoulli probabilities
    if model.b_graph_enc:
        graph_enc = model.graph_encoder(datum)
    else:
        graph_enc = None
    encoded_ld = model.encoder_ld(datum, addition=graph_enc)
    encoded_ud = model.encoder_ud(datum, addition=graph_enc)
    mega_logits = model.decoder(
        torch.stack([model.sampler(*encoded_ld) for _ in range(1000)]),
        torch.stack([model.sampler(*encoded_ud) for _ in range(1000)])).detach().numpy()
    mega_logits_mean = np.mean(mega_logits, axis=0)
    mega_mean = expit(mega_logits_mean)

    # Probabilities of y_true
    y_true = to_dense_adj(datum.edge_index)[0, ...].numpy()
    mega_logits_auc = roc_auc_score(y_true.flatten(), mega_logits_mean.flatten())
    # If p(x = 1) = y, probability of doing worse when x = 1 is y,
    # probability of doing worse when x = 0 is 1 - y
    log_prob = np.sum(np.log(y_true * mega_mean + (1 - y_true) * (1 - mega_mean)))

    if plots:
        import matplotlib.pyplot as plt
        # Confusion matrix with threshold 0.5
        print(confusion_matrix(y_true.flatten(), mega_mean.flatten() > 0.5, normalize="all"))
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(y_true)
        ax[0].set_title("True Adjacency")
        ax[1].imshow(mega_mean)
        ax[1].set_title("Model probability")
        ax[2].imshow((mega_mean > 0.5).astype(float))
        ax[2].set_title("Model probability thresholded 0.5")
        fig.tight_layout()
        fig.show()

    return mega_logits_auc, log_prob
