import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
import torch
import torch.optim as optim

class LinearWeightNorm(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_scale=None, weight_init_stdv=0.1):
        super(LinearWeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features) * weight_init_stdv)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if weight_scale is not None:
            assert type(weight_scale) == int
            self.weight_scale = Parameter(torch.ones(out_features, 1) * weight_scale)
        else:
            self.weight_scale = 1
    def forward(self, x):
        W = self.weight * self.weight_scale / torch.sqrt(torch.sum(self.weight ** 2, dim = 1, keepdim = True))
        return F.linear(x, W, self.bias)


class NetworkModel(nn.Module):
    def __init__(self, dim_in ,dim_out, hidden_dim=[1000,500,250,250,250],
                activations = [nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()]):
        super(NetworkModel, self).__init__()
        self.dim_in = dim_in
        self.num_hidden=len(hidden_dim)
        self.layers = torch.nn.ModuleList()
        self.hidden_size = 100
        self.n_layers = 1
        self.n_directions = 2 
        self.feature_dim = 100 * self.n_directions 

        self.gru_1 = nn.GRU(dim_in, self.hidden_size, self.n_layers, bidirectional=True)
        self.gru_2 = nn.GRU(200, 200, self.n_layers,bidirectional=True)
        self.fc1 = LinearWeightNorm(self.feature_dim, 200)
        self.fc2 = LinearWeightNorm(200, dim_out, weight_scale=1)
        self.activations = activations
    def __init__hidden(self, batch_size,hidden_size):  
        return torch.randn(self.n_layers * self.n_directions, batch_size, hidden_size)
    def forward(self, x):
        input = x.to(torch.float32)
        batch_size = input.size(0)
        input = torch.unsqueeze(input, dim = 0)
        hidden_0 = self.__init__hidden(batch_size,self.hidden_size)
        output, hidden = self.gru_1(input, hidden_0)
        if self.n_directions == 2:  
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]  
        fc_output = nn.Dropout()(self.fc1(hidden_cat))
        fc_output = self.activations[0](fc_output)
        self.fc_output = fc_output
        fc_output = self.fc2(fc_output)

        self.feature=hidden_cat
        return fc_output