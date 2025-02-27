from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
from Positional_encoding import LearnablePositionalEncoding
import math


class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(1.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


class FGN(nn.Module):
    def __init__(self, embed_size, feature_size, seq_length, hidden_size, layer_num=5,
                 hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01, lower_dim=12, fc_dim = 64):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.layer_num = layer_num
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.lower_dim = lower_dim
        self.fc_dim = fc_dim
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.encoder_hidden_size =32
       
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for _ in range(layer_num):
            self.weights.append(nn.Parameter(
                self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor)))
            self.biases.append(nn.Parameter(
                self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor)))

        self.embeddings_10 = nn.Parameter(torch.randn(90, self.lower_dim))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * lower_dim, self.fc_dim),
            nn.LeakyReLU(),
            nn.Linear(self.fc_dim, self.hidden_size)
        )
        self.gcn = GraphConvolution(in_features= self.hidden_size , out_features= self.hidden_size,dropout= 0.2)
        self.pos = LearnablePositionalEncoding(self.feature_size,self.seq_length)

        self.fc1 = nn.Linear(self.seq_length*(self.hidden_size ), self.seq_length*(self.hidden_size )//2)
        self.fc2 = nn.Linear(self.seq_length*(self.hidden_size )//2,self.seq_length*(self.hidden_size )//4)
        self.fc3 = nn.Linear(self.seq_length*(self.hidden_size )//4, 2)
        self.dim_reduction = nn.Linear(self.hidden_size,8)


        self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    def fourierGC(self, x, B, N, L):
        for i in range(self.layer_num):
            w_real, w_imag = self.weights[i][0], self.weights[i][1]
            b_real, b_imag = self.biases[i][0], self.biases[i][1]

            o_real = F.relu(
                torch.einsum('bli,ii->bli', x.real, w_real) - \
                torch.einsum('bli,ii->bli', x.imag, w_imag) + \
                b_real
            )
            o_imag = F.relu(
                torch.einsum('bli,ii->bli', x.imag, w_real) + \
                torch.einsum('bli,ii->bli', x.real, w_imag) + \
                b_imag
            )

            y = torch.stack([o_real, o_imag], dim=-1)
            y = F.softshrink(y, lambd=self.sparsity_threshold)

            if i == 0:
                x_new = y
            else:
                x_new = x_new + y

            x = torch.view_as_complex(x_new)

        return x

    def forward(self, x, adj):
        B, N, L = x.shape
        pos = self.pos(x)
        x = x.reshape(B, -1)
        x = self.tokenEmb(x)
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        x = x.reshape(B, (N*L)//2+1, self.frequency_size)
        bias = x
        x = self.fourierGC(x, B, N, L)
        x = x + bias
        x = x.reshape(B, (N*L)//2+1, self.embed_size)
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")
        x = x.reshape(B, N* L, self.embed_size)
        x = x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)
        x = torch.mul(x,pos)
        x = self.gcn(x,adj)
        x = x.reshape(B,-1)
        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))
        prediction_scores = self.fc3(fc2)

        return prediction_scores
