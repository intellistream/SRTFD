from torch import nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F
from torch.nn.functional import relu


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        x = x.float()
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class GCFAggMVC(nn.Module):
    def __init__(self, input_size, low_feature_dim, num_class, device):
        super(GCFAggMVC, self).__init__()
        self.encoders = Encoder(input_size, low_feature_dim).to(device)
        self.decoders = Decoder(input_size, low_feature_dim).to(device)

        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=low_feature_dim, nhead=1, dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(
            self.TransformerEncoderLayer, num_layers=1)

        self.fc = nn.Linear(low_feature_dim, num_class)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = x.to(torch.float32)
        z = self.encoders(x)
        xr = self.decoders(z)
        commonz = self.TransformerEncoderLayer(xr)
        commonz = self.TransformerEncoder(commonz)
        y = self.fc(commonz)
        return y

    def GCFAgg(self, xs):
        z = self.encoders(xs)
        commonz = torch.cat(z, 1)
        commonz, S = self.TransformerEncoderLayer(commonz)
        return commonz, S
