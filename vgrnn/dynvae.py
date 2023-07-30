"""VGRNN imlementation following https://arxiv.org/abs/1908.09710
Adapted for directed graphs
Extended for Gaussian mixture prior"""
import torch
from torch.nn import Sequential, Linear, ReLU, Softplus, ModuleList, Dropout
from torch_geometric.nn.conv import GCNConv, GINConv
from torch_geometric.nn import GIN
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

def get_nmode_prior(device, num_modes, latent_dim):
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

class Graph_GRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(Graph_GRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer
        
        # GRU weights
        self.weight_xz = ModuleList()
        self.weight_hz = ModuleList()
        self.weight_xr = ModuleList()
        self.weight_hr = ModuleList()
        self.weight_xh = ModuleList()
        self.weight_hh = ModuleList()
        
        for i in range(self.n_layer):
            _input_size = input_size if i == 0 else hidden_size
            self.weight_xz.append(GINConv(Linear(_input_size, hidden_size, bias=bias)))
            self.weight_hz.append(GINConv(Linear(hidden_size, hidden_size, bias=bias)))
            self.weight_xr.append(GINConv(Linear(_input_size, hidden_size, bias=bias)))
            self.weight_hr.append(GINConv(Linear(hidden_size, hidden_size, bias=bias)))
            self.weight_xh.append(GINConv(Linear(_input_size, hidden_size, bias=bias)))
            self.weight_hh.append(GINConv(Linear(hidden_size, hidden_size, bias=bias)))
    
    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size()).to(h.device)
        for i in range(self.n_layer):
            _input = inp if i == 0 else h_out[i - 1]
            z_g = torch.sigmoid(self.weight_xz[i](_input, edgidx) + self.weight_hz[i](h[i], edgidx))
            r_g = torch.sigmoid(self.weight_xr[i](_input, edgidx) + self.weight_hr[i](h[i], edgidx))
            h_tilde_g = torch.tanh(self.weight_xh[i](_input, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
            h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        out = h_out
        return out, h_out

class Encoder(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim, n_gin_layers=10):
        super().__init__()
        # GIN defaults to ReLU activation
        self.gin = GIN(in_channels=in_features, hidden_channels=hidden_dim,
                       num_layers=n_gin_layers, dropout=0.2)
        self.mean = torch.nn.Linear(hidden_dim, latent_dim)
        self.std = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.Softplus()
        )

    def forward(self, x, edge_list):
        hidden = self.gin(x, edge_list)
        return self.mean(hidden), self.std(hidden)
    
class PriorEncoder(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim, n_layers=1):
        super().__init__()
        # GIN defaults to ReLU activation
        self.dense =Sequential(
            Linear(in_features, hidden_dim), ReLU(),
            *[Sequential(Linear(hidden_dim, hidden_dim), Dropout(), ReLU()) for _ in range(n_layers)]
            )
        self.mean = torch.nn.Linear(hidden_dim, latent_dim)
        self.std = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.Softplus()
        )

    def forward(self, x):
        hidden = self.dense(x)
        return self.mean(hidden), self.std(hidden)

class InnerProductDecoder(torch.nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()
        
        self.act = act
        self.dropout = dropout
    
    def forward(self, inp_src, inp_dst):
        inp_src = F.dropout(inp_src, self.dropout, training=self.training)
        inp_dst = F.dropout(inp_dst, self.dropout, training=self.training)
        inp_dst_t = torch.transpose(inp_dst, dim0=0, dim1=1)
        x = torch.mm(inp_src, inp_dst_t)
        return self.act(x)

class DynVAE(torch.nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, eps, bias=False, n_prior_modes=0, device=None,
                 n_gin_layers=10, n_dense_layers=1, n_gru_layers=10, ):
        super(DynVAE, self).__init__()

        if device is None:
            device = torch.device("cpu")
        
        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_gru_layers = n_gru_layers
        
        # Functions for GRU recurrence
        self.phi_x = Sequential(Linear(x_dim, h_dim, bias=bias), ReLU())
        self.phi_z = Sequential(Linear(z_dim + z_dim, h_dim, bias=bias), ReLU())
        self.rnn = Graph_GRU(h_dim + h_dim, h_dim, n_gru_layers, bias)

        # Encoder: 2-layered GIN
        self.enc_src = Encoder(h_dim + h_dim, h_dim, z_dim, n_gin_layers=n_gin_layers)
        self.enc_dst = Encoder(h_dim + h_dim, h_dim, z_dim, n_gin_layers=n_gin_layers)

        # Prior: 2-layered MLP
        self.prior_src = PriorEncoder(h_dim, h_dim, z_dim, n_layers=n_dense_layers)
        self.prior_dst = PriorEncoder(h_dim, h_dim, z_dim, n_layers=n_dense_layers)

        # Prior Distribution
        self.simple_prior = n_prior_modes < 1
        if self.simple_prior:
            self.prior = dist.Normal(loc=torch.zeros(z_dim).to(device), scale=1.)
        else:
            self.prior = get_nmode_prior(device=device, num_modes=n_prior_modes, latent_dim=z_dim)
    
    def forward(self, x, edge_idx_list, adj_orig_dense_list, hidden_in=None):
        assert len(adj_orig_dense_list) == len(edge_idx_list)
        
        kld_loss = 0
        nll_loss = 0
        all_dec_t = []
        
        if hidden_in is None:
            h = torch.zeros(self.n_gru_layers, x.size(1), self.h_dim, requires_grad=True).to(x.device)
        else:
            h = hidden_in.clone().detach().requires_grad_(True).to(x.device)
        
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            #encoder
            enc_src_mean_t, enc_src_std_t = self.enc_src(torch.cat([phi_x_t, h[-1]], 1), edge_idx_list[t])
            enc_dst_mean_t, enc_dst_std_t = self.enc_dst(torch.cat([phi_x_t, h[-1]], 1), edge_idx_list[t])
            
            #prior
            prior_src_mean_t, prior_src_std_t = self.prior_src(h[-1])
            prior_dst_mean_t, prior_dst_std_t = self.prior_dst(h[-1])
            
            #sampling and reparameterization
            z_src_t = self._reparameterized_sample(enc_src_mean_t, enc_src_std_t)
            z_dst_t = self._reparameterized_sample(enc_dst_mean_t, enc_dst_std_t)
            phi_z_t = self.phi_z(torch.cat([z_src_t, z_dst_t], 1))
            
            #decoder
            dec_t = self.dec(z_src_t, z_dst_t)
            
            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), edge_idx_list[t], h)
            
            nnodes = adj_orig_dense_list[t].size()[0]

            enc_src_mean_t_sl = enc_src_mean_t[0:nnodes, :]
            enc_src_std_t_sl = enc_src_std_t[0:nnodes, :]

            enc_dst_mean_t_sl = enc_dst_mean_t[0:nnodes, :]
            enc_dst_std_t_sl = enc_dst_std_t[0:nnodes, :]

            prior_src_mean_t_sl = prior_src_mean_t[0:nnodes, :]
            prior_src_std_t_sl = prior_src_std_t[0:nnodes, :]

            prior_dst_mean_t_sl = prior_dst_mean_t[0:nnodes, :]
            prior_dst_std_t_sl = prior_dst_std_t[0:nnodes, :]

            dec_t_sl = dec_t[0:nnodes, 0:nnodes]
            
            #computing losses
            if self.simple_prior:
                kld_loss += self._kld_gauss(enc_src_mean_t_sl, enc_src_std_t_sl, prior_src_mean_t_sl, prior_src_std_t_sl) + \
                    self._kld_gauss(enc_dst_mean_t_sl, enc_dst_std_t_sl, prior_dst_mean_t_sl, prior_dst_std_t_sl)
            else:
                kld_loss += self._kld_gauss_zu(enc_src_mean_t_sl, enc_src_std_t_sl, prior_src_mean_t_sl, prior_src_std_t_sl) + \
                    self._kld_gauss_zu(enc_dst_mean_t_sl, enc_dst_std_t_sl, prior_dst_mean_t_sl, prior_dst_std_t_sl)
            nll_loss += self._nll_bernoulli(dec_t_sl, adj_orig_dense_list[t])

            all_dec_t.append(dec_t_sl)
        
        return kld_loss, nll_loss, None, None, None, all_dec_t
    
    def dec(self, z_src, z_dst):
        outputs = InnerProductDecoder(act=lambda x:x)(z_src, z_dst)
        return outputs
    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
     
    def _init_weights(self, stdv):
        pass
    
    def _reparameterized_sample(self, mean, std):
        epsilon = self.prior.sample((mean.size(0),))
        z = mean + std * epsilon
        return z
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        # See: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)
    
    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        # https://arxiv.org/abs/1312.6114
        std_log = torch.log(std_in + self.eps)
        kld_element =  torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                            torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element
    
    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits,
                                                          target=target_adj_dense,
                                                          pos_weight=posw,
                                                          reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0,1])
        return - nll_loss
