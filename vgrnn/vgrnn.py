"""VGRNN imlementation following https://arxiv.org/abs/1908.09710"""
import torch
from torch.nn import Sequential, Linear, ReLU, Softplus
from torch_geometric.nn import Sequential as GeomSequential
from torch_geometric.nn.models import GCN, GraphSAGE, GIN
from torch_geometric.nn.conv import GCNConv, SAGEConv, GINConv
from torch.autograd import Variable
import torch.nn.functional as F

class Graph_GRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(Graph_GRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer
        
        # GRU weights
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        
        for i in range(self.n_layer):
            _input_size = input_size if i == 0 else hidden_size
            self.weight_xz.append(GINConv(Linear(_input_size, hidden_size, bias=bias)))
            self.weight_hz.append(GINConv(Linear(hidden_size, hidden_size, bias=bias)))
            self.weight_xr.append(GINConv(Linear(_input_size, hidden_size, bias=bias)))
            self.weight_hr.append(GINConv(Linear(hidden_size, hidden_size, bias=bias)))
            self.weight_xh.append(GINConv(Linear(_input_size, hidden_size, bias=bias)))
            self.weight_hh.append(GINConv(Linear(hidden_size, hidden_size, bias=bias)))
    
    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size())
        for i in range(self.n_layer):
            _input = inp if i == 0 else h_out[i - 1]
            z_g = torch.sigmoid(self.weight_xz[i](_input, edgidx) + self.weight_hz[i](h[i], edgidx))
            r_g = torch.sigmoid(self.weight_xr[i](_input, edgidx) + self.weight_hr[i](h[i], edgidx))
            h_tilde_g = torch.tanh(self.weight_xh[i](_input, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
            h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        out = h_out
        return out, h_out

class InnerProductDecoder(torch.nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()
        
        self.act = act
        self.dropout = dropout
    
    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)

class VGRNN(torch.nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, eps, bias=False):
        super(VGRNN, self).__init__()
        
        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        
        # Functions for GRU recurrence
        self.phi_x = Sequential(Linear(x_dim, h_dim, bias=bias), ReLU())
        self.phi_z = Sequential(Linear(z_dim, h_dim, bias=bias), ReLU())
        self.rnn = Graph_GRU(h_dim + h_dim, h_dim, n_layers, bias)

        # Encoder: 2-layered GIN
        self.enc = GINConv(Sequential(Linear(h_dim + h_dim, h_dim), ReLU()))
        self.enc_mean = GCNConv(h_dim, z_dim)
        self.enc_std = GINConv(Sequential(Linear(h_dim, z_dim), Softplus()))

        # Prior: 2-layered MLP
        self.prior = Sequential(Linear(h_dim, h_dim), ReLU())
        self.prior_mean = Sequential(Linear(h_dim, z_dim))
        self.prior_std = Sequential(Linear(h_dim, z_dim), Softplus())
    
    def forward(self, x, edge_idx_list, adj_orig_dense_list, hidden_in=None):
        assert len(adj_orig_dense_list) == len(edge_idx_list)
        
        kld_loss = 0
        nll_loss = 0
        all_enc_mean, all_enc_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_dec_t, all_z_t = [], []
        
        if hidden_in is None:
            h = torch.zeros(self.n_layers, x.size(1), self.h_dim, requires_grad=True).to(x.device)
        else:
            h = torch.tensor(hidden_in, requires_grad=True).to(x.device)
        
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])
            
            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1), edge_idx_list[t])
            enc_mean_t = self.enc_mean(enc_t, edge_idx_list[t])
            enc_std_t = self.enc_std(enc_t, edge_idx_list[t])
            
            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            
            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)
            
            #decoder
            dec_t = self.dec(z_t)
            
            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), edge_idx_list[t], h)
            
            nnodes = adj_orig_dense_list[t].size()[0]
            enc_mean_t_sl = enc_mean_t[0:nnodes, :]
            enc_std_t_sl = enc_std_t[0:nnodes, :]
            prior_mean_t_sl = prior_mean_t[0:nnodes, :]
            prior_std_t_sl = prior_std_t[0:nnodes, :]
            dec_t_sl = dec_t[0:nnodes, 0:nnodes]
            
            #computing losses
#             kld_loss += self._kld_gauss_zu(enc_mean_t, enc_std_t)
            kld_loss += self._kld_gauss(enc_mean_t_sl, enc_std_t_sl, prior_mean_t_sl, prior_std_t_sl)
            nll_loss += self._nll_bernoulli(dec_t_sl, adj_orig_dense_list[t])
            
            all_enc_std.append(enc_std_t_sl)
            all_enc_mean.append(enc_mean_t_sl)
            all_prior_mean.append(prior_mean_t_sl)
            all_prior_std.append(prior_std_t_sl)
            all_dec_t.append(dec_t_sl)
            all_z_t.append(z_t)
        
        return kld_loss, nll_loss, all_enc_mean, all_prior_mean, h, all_dec_t
    
    def dec(self, z):
        outputs = InnerProductDecoder(act=lambda x:x)(z)
        return outputs
    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
     
    def _init_weights(self, stdv):
        pass
    
    def _reparameterized_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1).to(mean.device)
        return eps1.mul(std).add_(mean)
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)
    
    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
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
