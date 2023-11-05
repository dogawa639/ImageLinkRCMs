import torch
from torch import nn, tensor
from torch.nn import functional as F

from models.general import Softplus, SLN
from models.transformer import MultiHeadSelfAttention, TransformerEncoder

__all__ = ["GAT", "GT"]


# https://arxiv.org/abs/1710.10903
class GATBlock(nn.Module):
    # input: (*, node_num, emb_dim), adj_matrix: (node_num, node_num)
    # output: (*, node_num, emb_dim)
    def __init__(self, emb_dim, adj_matrix, in_emb_dim=None, num_head=1, dropout=0.0, sn=False, output_atten=False, atten_fn="matmul"):
        super().__init__()
        kwargs = {"in_emb_dim": in_emb_dim, "num_head": num_head, "dropout": dropout, "output_atten": output_atten, "sn": sn, "atten_fn": atten_fn}
        self.attention = MultiHeadSelfAttention(emb_dim, **kwargs)
        self.adj_matrix = adj_matrix

    def forward(self, x):
        # x: (*, node_num, emb_dim)
        n = x.shape[-2]
        return x + self.attention(x, self.adj_matrix.expand(*x.shape[:-2], n, n))


class GAT(nn.Module):
    def __init__(self, emb_dim_in, emb_dim_out, adj_matrix, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, output_atten=False, sn=False, sln=False, w_dim=None, atten_fn="dense"):
        super().__init__()
        self.emb_dim_in = emb_dim_in
        self.emb_dim_out = emb_dim_out
        self.in_emb_dim = in_emb_dim if in_emb_dim is not None else emb_dim_out
        self.num_head = num_head
        self.dropout = dropout
        self.depth = depth
        self.output_atten = output_atten
        self.sln = sln
        self.w_dim = w_dim
        self.atten_fn = atten_fn

        self.adj_matrix = adj_matrix

        self.f0 = nn.Linear(emb_dim_in, emb_dim_out, bias=False)
        kwargs = {"in_emb_dim": in_emb_dim, "num_head": num_head, "dropout": dropout, "output_atten": output_atten, "sn": sn, "atten_fn": atten_fn}
        self.gat_blocks = nn.ModuleList([GATBlock(emb_dim_out, adj_matrix, **kwargs)] + [GATBlock(emb_dim_out, adj_matrix, **kwargs) for _ in range(depth - 1)])
        
        if not sln:
            self.norms = nn.ModuleList([nn.LayerNorm(emb_dim_out) for _ in range(depth)])
        else:
            self.norms = nn.ModuleList([SLN(w_dim, emb_dim_out) for _ in range(depth)])

    def forward(self, x, w=None):
        atten = None
        x = self.f0(x)
        if self.sln:
            for norm in self.norms:
                norm.set_w(w)
        for gat_block in self.gat_blocks:
            x, atten_agg = gat_block(x)
            x = self.norms[i](x)
            if atten is None:
                atten = atten_agg
            else:
                atten = torch.matmul(atten_agg, atten)
        return x, atten
    

# https://arxiv.org/abs/2012.09699
class GT(nn.Module):
    # input: (bs, node_num, emb_dim_in) or (node_num, emb_dim_in), pos_encoding: (node_num, enc_dim)
    # output: (bs, node_num, emb_dim_out) or (node_num, emb_dim_out)
    def __init__(self, emb_dim_in, emb_dim_out, adj_matrix, enc_dim=3, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, pre_norm=False, sn=False, sln=False, w_dim=None, output_atten=False):
        # k: number of heads
        super().__init__()
        self.emb_dim_in = emb_dim_in
        self.emb_dim_out = emb_dim_out
        self.enc_dim = enc_dim
        self.in_emb_dim = in_emb_dim if in_emb_dim is not None else emb_dim_out
        self.num_head = num_head
        self.dropout = dropout
        self.depth = depth
        self.output_atten = output_atten

        self.adj_matrix = adj_matrix

        self.f0 = nn.Linear(emb_dim_in, emb_dim_out, bias=False)
        kwargs = {"enc_dim": enc_dim, "in_emb_dim": in_emb_dim, "num_head": num_head, "dropout": dropout, "depth": depth, "pre_norm": pre_norm, "output_atten": output_atten, "sn": sn, "sln": sln, "w_dim": w_dim}
        self.transformer = TransformerEncoder(emb_dim_out, **kwargs)

        self.adj_matrix = adj_matrix
        self.e, self.v = torch.linalg.eigh(adj_matrix)

    def forward(self, x, w=None):
        x = self.f0(x)
        enc=self.v[:, :self.enc_dim]
        if x.dim() == 3:
            enc = enc.expand(x.shape[0], *enc.shape)
        return self.transformer(x, enc=enc, w=w)

    
