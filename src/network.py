
"""
This code is the model structure of DDA-GTN with encoder and decoder.
"""


import torch

from model import FastGTNs
from torch import nn


class PygGTNs_LP(torch.nn.Module):
    def __init__(self, args, num_edge_type, node_features, num_nodes):
        super().__init__()
        self.encoder = FastGTNs(num_edge_type=num_edge_type,
                                w_in=node_features.shape[1],
                                num_nodes=num_nodes,
                                num_class=3,
                                args=args)

        self.w = nn.Sequential(
            nn.BatchNorm1d(2 * args.node_dim),
            nn.Linear(2 * args.node_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),
        )
        self.input_norm = nn.LayerNorm(node_features.shape[1])


    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.01)

    def encode(self, node_features, A, num_nodes):
        x = self.input_norm(node_features)
        x = self.encoder(A, x, num_nodes=num_nodes)
        return x

    def decode(self, z, edge_label_index):

        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]

        c = torch.cat((src, dst), dim=-1)
        r = self.w(c)

        return r

    def forward(self, node_features, A, num_nodes, edge_label_index):
        z = self.encode(node_features, A, num_nodes)
        out = self.decode(z, edge_label_index)
        return out

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

