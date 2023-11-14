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
    def __init__(self, emb_dim, adj_matrix, 
                 in_emb_dim=None, num_head=1, dropout=0.0, sn=False, output_atten=False, atten_fn="matmul"):
        super().__init__()
        self.output_atten = output_atten
        kwargs = {
            "in_emb_dim": in_emb_dim, 
            "num_head": num_head, 
            "dropout": dropout, 
            "output_atten": output_atten, 
            "sn": sn, 
            "atten_fn": atten_fn
            }
        self.attention = MultiHeadSelfAttention(emb_dim, **kwargs)
        self.adj_matrix = nn.parameter.Parameter(adj_matrix, requires_grad=False)

    def forward(self, x):
        # x: (*, node_num, emb_dim)
        n = x.shape[-2]
        y = self.attention(x, self.adj_matrix.expand(*x.shape[:-2], n, n))
        if self.output_atten:
            return x + y[0], y[1]
        return x + y


class GAT(nn.Module):
    def __init__(self, emb_dim_in, emb_dim_out, adj_matrix, 
                 in_emb_dim=None, num_head=1, dropout=0.0, depth=1, sn=False, sln=False, w_dim=None, output_atten=False, atten_fn="dense"):
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

        self.adj_matrix = nn.parameter.Parameter(adj_matrix, requires_grad=False)

        self.f0 = nn.Linear(emb_dim_in, emb_dim_out, bias=False)
        kwargs = {
            "in_emb_dim": in_emb_dim, 
            "num_head": num_head, 
            "dropout": dropout, 
            "output_atten": output_atten, 
            "sn": sn, 
            "atten_fn": 
            atten_fn
            }
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
        for i, gat_block in enumerate(self.gat_blocks):
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
    def __init__(self, emb_dim_in, emb_dim_out, adj_matrix, 
                 enc_dim=3, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, pre_norm=False, sn=False, sln=False, w_dim=None, output_atten=False):
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

        self.f0 = nn.Linear(emb_dim_in, emb_dim_out, bias=False)
        kwargs = {
            "enc_dim": enc_dim, 
            "in_emb_dim": in_emb_dim, 
            "num_head": num_head, 
            "dropout": dropout, 
            "depth": depth, 
            "pre_norm": pre_norm, 
            "output_atten": output_atten, 
            "sn": sn,
            "sln": sln, 
            "w_dim": w_dim
            }
        self.transformer = TransformerEncoder(emb_dim_out, **kwargs)

        self.adj_matrix = nn.parameter.Parameter(adj_matrix.to(torch.float32).to_dense(), requires_grad=False)
        self.e, self.v = torch.linalg.eigh(adj_matrix.to("cpu").to_dense())
        self.e = nn.parameter.Parameter(self.e.to(adj_matrix.device), requires_grad=False)
        self.v = nn.parameter.Parameter(self.v.to(adj_matrix.device), requires_grad=False)

    def forward(self, x, w=None):
        x = self.f0(x)
        enc = self.v[:, :self.enc_dim]
        if x.dim() == 3:
            enc = enc.expand(x.shape[0], *enc.shape)
        return self.transformer(x, enc=enc, w=w)


# test
if __name__ == "__main__":
    device = "mps"
    bs = 3
    emb_dim_in = 4
    emb_dim_out = 5
    node_num = 6
    adj_matrix = torch.randint(0, 2, (node_num, node_num)).to(torch.float32)
    w_dim = 7

    gat = GAT(emb_dim_in, emb_dim_out, adj_matrix,
                 in_emb_dim=None, num_head=2, dropout=0.1, depth=2, output_atten=True, sn=True, sln=False, w_dim=w_dim, atten_fn="dense").to(device)
    x = torch.randn((bs, node_num, emb_dim_in), device=device)
    w = torch.randn((bs, w_dim), device=device)

    out1, atten1 = gat(x, w)
    print(out1.shape, atten1.shape)

    gt = GT(emb_dim_in, emb_dim_out, adj_matrix,
                    enc_dim=3, in_emb_dim=None, num_head=2, dropout=0.0, depth=2, pre_norm=True, sn=True, sln=True, w_dim=w_dim, output_atten=True).to(device)
    out2, attens2 = gt(x, w)
    print(out2.shape, attens2[0].shape)

