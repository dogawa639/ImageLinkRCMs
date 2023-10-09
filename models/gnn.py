import torch
from torch import nn, tensor
from torch.nn import functional as F

from .general import Softplus

__all__ = ["GAT", "GATBlock"]


class GATBlock(nn.Module):
    # input: (node_num, in_channel), adj_matrix: (node_num, node_num)
    # output: (node_num, out_channel)
    def __init__(self, in_channel, out_channel, k=1, dropout=0.0, residual=True):
        # k: number of heads
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.k = k
        self.residual = residual

        self.layer_norm1 = nn.LayerNorm(self.in_channel)
        self.layer_norm2 = nn.LayerNorm(self.in_channel)
        self.dropout = nn.Dropout(p=dropout)

        self.ff0 = [nn.Linear(self.in_channel, self.in_channel, bias=False) for _ in range(self.k)]
        self.ff1 = [nn.Linear(self.in_channel, self.out_channel, bias=False) for _ in range(self.k)]
        self.attn_fc = [nn.Linear(2 * self.in_channel, 1, bias=False) for _ in range(self.k)]

    def forward(self, x, adj):
        n = x.shape[0]
        x_norm = self.layer_norm1(x)
        h_next = torch.zeros((n, self.out_channel), device=x.device)
        for i in range(self.k):
            h = self.ff0[i](x_norm)  # (node_num, in_channel)

            h_cross = torch.cat((h.expand(n, n, self.in_channel).transpose(0, 1), h.expand(n, n, self.in_channel)), 2)  # (node_num, node_num, 2*in_channel) 行ベクタ | 列ベクタ
            e = F.leaky_relu(self.attn_fc[i](h_cross).squeeze(2))

            adj = adj + torch.eye(n, device=adj.device)
            e = torch.where(adj > 0, e, tensor(-9e15, dtype=torch.float32, device=adj.device))
            atten = F.softmax(e, dim=1) # (node_num, node_num)
            atten = self.dropout(atten)

            attentioned = torch.matmul(atten, h)
            if self.residual:
                attentioned = attentioned + x
            attentioned = self.layer_norm2(attentioned)

            h_next = h_next + F.elu(self.ff1[i](attentioned)) / self.k  # (node_num, out_channel)
        return h_next



class GAT(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, dropout=0.0, depth=1, residual=True):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.k = k
        self.dropout = dropout
        self.depth = depth
        self.residual = residual

        kwargs = {"k": k, "dropout": dropout, "residual": residual}
        self.gat_blocks = nn.ModuleList([GATBlock(in_channel, out_channel, **kwargs)] + [GATBlock(out_channel, out_channel, **kwargs) for _ in range(depth - 1)])

    def forward(self, x, adj):
        for gat_block in self.gat_blocks:
            x = gat_block(x, adj)
        return x