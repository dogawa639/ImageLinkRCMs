import torch
from torch import nn, tensor
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from general import Softplus, SLN, FF

__all__ = ["MultiHeadAttention", "MultiHeadSelfAttention", "TransformerEncoder", "TransformerDecoder"]


class MultiHeadAttention(nn.Module):
    # query: (*, input_channel, emb_dim), key: (*, source_channel, emb_dim), value: (*, source_channel, emb_dim)
    # output: (*, input_channel, emb_dim)
    # softmax(QK^T/sqrt(d_k))V
    def __init__(self, emb_dim, in_emb_dim=None, num_head=1, dropout=0.0,  sn=False, output_atten=False, atten_fn="matmul"):
        # atten_fn: "matmul" or "dense"
        super().__init__()
        self.emb_dim = emb_dim
        self.in_emb_dim = in_emb_dim if in_emb_dim is not None else emb_dim
        self.num_head = num_head
        self.dropout = dropout
        self.output_atten = output_atten
        self.atten_fn = atten_fn

        self.dropout = nn.Dropout(p=dropout)

        if atten_fn == "dense":
            self.dense_atten = nn.ModuleList([nn.Linear(emb_dim, 1, bias=False) for _ in range(num_head)])  # emb_dim -> 1

        if sn:
            self.q_dense = nn.ModuleList([spectral_norm(nn.Linear(emb_dim, self.in_emb_dim, bias=False)) for _ in range(num_head)])
            self.k_dense = nn.ModuleList([spectral_norm(nn.Linear(emb_dim, self.in_emb_dim, bias=False)) for _ in range(num_head)])
            self.v_dense = nn.ModuleList([spectral_norm(nn.Linear(emb_dim, self.in_emb_dim, bias=False)) for _ in range(num_head)])

            self.last_dense = spectral_norm(nn.Linear(self.in_emb_dim * num_head, emb_dim, bias=False))
        else:
            self.q_dense = nn.ModuleList([nn.Linear(emb_dim, self.in_emb_dim, bias=False) for _ in range(num_head)])
            self.k_dense = nn.ModuleList([nn.Linear(emb_dim, self.in_emb_dim, bias=False) for _ in range(num_head)])
            self.v_dense = nn.ModuleList([nn.Linear(emb_dim, self.in_emb_dim, bias=False) for _ in range(num_head)])

            self.last_dense = nn.Linear(self.in_emb_dim * num_head, emb_dim, bias=False)

    def forward(self, q, k, v, mask=None):
        # mask: None or (*, source_channel) or (*, input_channel, source_channel)
        input_channel = q.shape[-2]
        source_channel = k.shape[-2]
        if source_channel != v.shape[-2]:
            raise Exception("source_channel and value_channel should be the same")
        atten_agg = None
        h_tmp = None
        for i in range(self.num_head):
            in_q = self.q_dense[i](q)  # (*, input_channel, in_emb_dim)
            in_k = self.k_dense[i](k)  # (*, source_channel, in_emb_dim)
            in_v = self.v_dense[i](v)  # (*, source_channel, in_emb_dim)

            logits = self.atten_mechanism(in_q, in_k, i)  # (*, input_channel, source_channel)
            if mask is not None:
                if mask.dim() == atten.dim():
                    logits = logits * mask
                else:
                    logits = logits * mask.unsqueeze(-2)
            logits = torch.where(logits != 0, logits, torch.full_like(logits, -9e15))
            atten = F.softmax(logits, dim=-1)  # (*, input_channel, source_channel)
            if atten_agg is None:
                atten_agg = atten.clone().detach() / self.num_head
            else:
                atten_agg = atten_agg + atten.clone().detach() / self.num_head
            atten = self.dropout(atten)
            attentioned = torch.matmul(atten, in_v)  # (*, input_channel, in_emb_dim)
            if h_tmp is None:
                h_tmp = attentioned
            else:
                h_tmp = torch.cat((h_tmp, attentioned), dim=-1)
        h = self.last_dense(h_tmp)
        if self.output_atten:
            return h, atten_agg
        else:
            return h
        
    def atten_mechanism(self, in_q, in_k, head):
        if self.atten_fn == "matmul":
            logits = torch.matmul(in_q, in_k.transpose(-2, -1)) / (in_k.shape[-2] ** 0.5)
        elif self.atten_fn == "dense":
            f1 = self.dense_atten[head](in_q).unsqueeze(-2)  # (*, input_channel, 1, 1)
            f2 = self.dense_atten[head](in_k).unsqueeze(-3)  # (*, 1, source_channel, 1)
            logits = f1 + f2
        return logits

    
class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, mask=None):
        return super().forward(x, x, x, mask=mask)
    

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, in_emb_dim=None, num_head=1, dropout=0.0, pre_norm=False, sn=False, sln=False, w_dim=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.in_emb_dim = in_emb_dim
        self.num_head = num_head
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.sln = sln
        self.w_dim = w_dim

        if not sln:
            self.layer_norm1 = nn.LayerNorm(emb_dim)
            self.layer_norm2 = nn.LayerNorm(emb_dim)
        else:
            if w_dim is None:
                raise Exception("w_dim should be specified when sln is True")
            self.layer_norm1 = SLN(w_dim, emb_dim)
            self.layer_norm2 = SLN(w_dim, emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.attention = MultiHeadAttention(emb_dim, in_emb_dim=in_emb_dim, num_head=num_head, dropout=dropout, output_atten=True, sn=sn)  # output: (*, input_channel, emb_dim)
        self.ff = FF(emb_dim, emb_dim, emb_dim*2, sn=sn)

        self.w = None

    def forward(self, q, kv=None, mask=None):
        # q: (bs, input_channel, emb_dim)
        # kv: None or (bs, source_channel, emb_dim)
        if self.sln and self.w is None:
            raise Exception("w should be specified when using sln")

        if self.pre_norm:
            q = self.layer_norm1(q)
            if kv is None:
                h1, atten = self.attention(q, q, q, mask=mask)  # (bs, input_channel, emb_dim)
            else:
                h1, atten = self.attention(q, kv, kv, mask=mask)
            h1 = h1 + q

            h1 = self.dropout(self.layer_norm2(h1))
            h2 = self.ff(h1)
            h = h2 + h1
        else:
            if kv is None:
                h1, atten = self.attention(q, q, q, mask=mask)
            else:
                h1, atten = self.attention(q, kv, kv, mask=mask)
            h1 = h1 + q
            h1 = self.layer_norm1(h1)

            h1 = self.dropout(h1)
            h2 = self.ff(h1)
            h = h2 + h1
            h = self.layer_norm2(h)
        return h, atten
    
    def set_w(self, w):
        # w: (bs, w_dim) or (w_dim)
        self.w = w
        if self.sln:
            self.layer_norm1.set_w(w)
            self.layer_norm2.set_w(w)
        else:
            pass


class TransformerEncoder(nn.Module):
    # input, output: (bs, input_channel, emb_dim)
    def __init__(self, emb_dim, enc_dim=None, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, pre_norm=False, sn=False, sln=False, w_dim=None, output_atten=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_dim = enc_dim
        self.in_emb_dim = in_emb_dim
        self.num_head = num_head
        self.depth = depth
        self.pre_norm = pre_norm
        self.sn = sn
        self.sln = sln
        self.w_dim = w_dim
        self.output_atten = output_atten

        kwargs = {"in_emb_dim": in_emb_dim, "num_head": num_head, "dropout": dropout, "pre_norm": pre_norm, "sn": sn, "sln": sln, "w_dim": w_dim}
        self.tf_blocks = nn.ModuleList([TransformerBlock(emb_dim, **kwargs) for _ in range(depth)])
        if enc_dim is not None:
            self.ff_enc = FF(enc_dim, emb_dim, enc_dim*2, bias=False)

    def forward(self, x, enc, mask=None, w=None):
        # x: (bs, input_channel, emb_dim)
        # enc: None or (bs, input_channel, enc_dim)
        # w: None or (bs, w_dim) or (w_dim)
        if x.dim() != 3:
            raise Exception("x should be 3 dim")
        if enc is not None:
            x = x + self.ff_enc(enc)
        attens = []
        for tf_block in self.tf_blocks:
            tf_block.set_w(w)
            x, atten = tf_block(x, mask=mask)
            attens.append(atten)
        return x, attens
    
class TransformerDecoder(nn.Module):
    # q: (bs, input_channel, emb_dim)
    # k, v: (bs, source_channel, enc_dim)
    def __init__(self, emb_dim, enc_dim=None, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, pre_norm=False, sn=False, sln=False, w_dim=None, output_atten=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_dim = enc_dim
        self.in_emb_dim = in_emb_dim
        self.num_head = num_head
        self.depth = depth
        self.pre_norm = pre_norm
        self.sn = sn
        self.sln = sln
        self.w_dim = w_dim
        self.output_atten = output_atten

        kwargs = {"in_emb_dim": in_emb_dim, "num_head": num_head, "dropout": dropout, "sn": sn}
        self.self_attentions = nn.ModuleList([MultiHeadSelfAttention(emb_dim, **kwargs) for _ in range(depth)])
        if not sln:
            self.norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(depth)])
        else:
            if w_dim is None:
                raise Exception("w_dim should be specified when sln is True")
            self.norms = nn.ModuleList([SLN(w_dim, emb_dim) for _ in range(depth)])

        kwargs = {"in_emb_dim": in_emb_dim, "num_head": num_head, "dropout": dropout, "pre_norm": pre_norm, "sn": sn, "sln": sln, "w_dim": w_dim}
        self.tf_blocks = nn.ModuleList([TransformerBlock(emb_dim, **kwargs) for _ in range(depth)])

        if enc_dim is not None:
            self.ff_enc = FF(enc_dim, emb_dim, enc_dim*2, bias=False)

    def forward(self, x, enc, kv, mask=None, w=None):
        # x: (bs, input_channel, emb_dim)
        # enc: None or (bs, input_channel, enc_dim)
        # kv: (bs, source_channel, enc_dim)
        # w: None or (bs, w_dim) or (w_dim)
        if x.dim() != 3:
            raise Exception("x should be 3 dim")
        if enc is not None:
            x = x + self.ff_enc(enc)
        attens = []
        for i in range(self.depth):
            if self.sln:
                self.norms[i].set_w(w)
                self.tf_blocks[i].set_w(w)
            if self.pre_norm:
                x = self.norms[i](x)
                h, _ = self.self_attentions[i](x, mask=mask)
                x = x + h
                x, atten = self.tf_blocks[i](x, kv, mask=mask)
                attens.append(atten)
            else:
                h, _ = self.self_attentions[i](x, mask=mask)
                x = x + h
                x = self.norms[i](x)
                x, atten = self.tf_blocks[i](x, kv, mask=mask)
                attens.append(atten)
        return x, attens



    

