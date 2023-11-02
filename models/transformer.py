import torch
from torch import nn, tensor
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from general import Softplus, SLN

__all__ = ["AttentionBlock", "Transformer"]


class AttentionBlock(nn.Module):
    # input: (node_num, in_channel), pos_encoding: (node_num, enc_dim)
    # output: (node_num, out_channel)
    def __init__(self, in_channel, out_channel, enc_dim, k=1, dropout=0.0, residual=True, sn=False, sln=False, w_dim=None):
        # k: number of heads
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.enc_dim = enc_dim
        self.k = k
        self.residual = residual
        self.sn = sn
        self.sln = sln
        self.w_dim = w_dim

        if not sln:
            self.layer_norm1 = nn.LayerNorm(self.in_channel)
            self.layer_norm2 = nn.LayerNorm(self.in_channel)
        else:
            if w_dim is None:
                raise Exception("w_dim should be specified when sln is True")
            self.layer_norm1 = SLN(self.w_dim, self.in_channel)
            self.layer_norm2 = SLN(self.w_dim, self.in_channel)
        self.dropout = nn.Dropout(p=dropout)

        if self.enc_dim is not None:
            self.ff_enc = nn.Linear(self.enc_dim, self.in_channel, bias=True)
        if not sn:
            self.ff0 = nn.ModuleList([nn.Linear(self.in_channel, self.in_channel, bias=False) for _ in range(self.k)])
            self.ff1 = nn.ModuleList([nn.Linear(self.in_channel, self.out_channel, bias=False) for _ in range(self.k)])
            self.attn_fc = nn.ModuleList([nn.Linear(2 * self.in_channel, 1, bias=False) for _ in range(self.k)])
        else:
            self.ff0 = nn.ModuleList([spectral_norm(nn.Linear(self.in_channel, self.in_channel, bias=False)) for _ in range(self.k)])
            self.ff1 = nn.ModuleList([spectral_norm(nn.Linear(self.in_channel, self.out_channel, bias=False)) for _ in range(self.k)])
            self.attn_fc = nn.ModuleList([spectral_norm(nn.Linear(2 * self.in_channel, 1, bias=False)) for _ in range(self.k)])

    def forward(self, x, enc=None, w=None):
        # x: (bs, link_num, in_channel)
        # enc: (bs, link_num, enc_dim) positional encoding
        # w: (bs, w_dim)
        if enc is not None and x.shape[0] != enc.shape[0]:
            raise Exception("x and enc should have the same batch size")
        if w is not None and x.shape[0] != w.shape[0]:
            raise Exception("x and w should have the same batch size")
        bs = x.shape[0]
        n = x.shape[1]
        if enc is not None:
            x = x + self.ff_enc(enc)  # (bs, link_num, in_channel)
        if self.sln:
            w = w.unsqueeze(1).expand(bs, n, -1)
            x_norm = self.layer_norm1(x, w)
        else:
            x_norm = self.layer_norm1(x)
        h_next = torch.zeros((bs, n, self.out_channel), device=x.device)
        atten_agg = None
        for i in range(self.k):
            h = self.ff0[i](x_norm)  # (bs, node_num, in_channel)
            h_expand = h.unsqueeze(2).expand(bs, n, n, self.in_channel)  # (bs, node_num, 1, in_channel) -> (bs, node_num, node_num, in_channel)

            h_cross = torch.cat((h_expand, h_expand.transpose(1, 2)), 3)  # (bs, node_num, node_num, 2*in_channel) 行ベクタ | 列ベクタ
            e = F.leaky_relu(self.attn_fc[i](h_cross).squeeze(3))

            atten = F.softmax(e, dim=2)  # (bs, node_num, node_num)
            atten = self.dropout(atten)

            attentioned = torch.matmul(atten, h)
            if self.residual:
                attentioned = attentioned + x
            if self.sln:
                attentioned = self.layer_norm2(attentioned, w)
            else:
                attentioned = self.layer_norm2(attentioned)

            h_next = h_next + F.elu(self.ff1[i](attentioned)) / self.k  # (bs, node_num, out_channel)
            if atten_agg is None:
                atten_agg = atten.clone().detach() / self.k
            else:
                atten_agg = atten_agg + atten.clone().detach() / self.k
        return h_next, atten_agg


class Transformer(nn.Module):
    # input: (bs, link_num, input_channel)
    # output: (bs, link_num, output_channel)
    def __init__(self, in_channel, out_channel, enc_dim=None, k=1, dropout=0.0, depth=1, residual=True, sn=False, sln=False, w_dim=None):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.enc_dim = enc_dim
        self.k = k
        self.dropout = dropout
        self.depth = depth
        self.residual = residual
        self.sn = sn
        self.sln = sln
        self.w_dim = w_dim

        kwargs = {"enc_dim": enc_dim, "k": k, "dropout": dropout, "residual": residual, "sn": sn, "sln": sln, "w_dim": w_dim}
        self.at_blocks = nn.ModuleList([AttentionBlock(in_channel, out_channel, **kwargs)] + [AttentionBlock(out_channel, out_channel, **kwargs) for _ in range(depth - 1)])

    def forward(self, x, enc, w=None):
        # x: (bs, link_num, input_channel)
        # enc: None or (bs, link_num, enc_dim)
        # w: None or (bs, w_dim)
        if x.dim() != 3:
            raise Exception("x should be 3 dim")
        atten = None
        for i, at_block in enumerate(self.at_blocks):
            if i == 0:
                x, atten_agg = at_block(x, enc, w)
            else:
                x, atten_agg = at_block(x, w=w)
            if atten is None:
                atten = atten_agg
            else:
                atten = torch.einsum("bij, bjk -> bik", atten_agg, atten)
        return x, atten
    

