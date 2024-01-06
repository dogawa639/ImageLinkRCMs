if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from models.general import *
    from models.transformer import *

    device = "mps"
    bs = 3
    emb_dim = 4
    enc_dim = 5
    w_dim = 6
    seq_len_source = 10
    seq_len_input = 5

    transformer_enc = TransformerEncoder(emb_dim, enc_dim=enc_dim, num_head=2, depth=3, dropout=0.1, pre_norm=True,
                                         sn=True, sln=True, w_dim=w_dim, output_atten=True).to(device)
    x = torch.randn((bs, seq_len_source, emb_dim), device=device)
    enc_sou = torch.randn((bs, seq_len_source, enc_dim), device=device)  # source sequence
    enc_inp = torch.randn((bs, seq_len_input, enc_dim), device=device)  # input sequence
    mask_sou = (torch.randn((bs, seq_len_source), device=device) > 0.0).to(torch.float32)
    mask_inp = (torch.randn((bs, seq_len_input, seq_len_input), device=device) > 0.0).to(torch.float32)
    w = torch.randn((bs, w_dim), device=device)
    q = torch.randn((bs, seq_len_input, emb_dim), device=device)

    kv, attens_sou = transformer_enc(x, enc_sou, mask=mask_sou, w=w)

    transformer_dec = TransformerDecoder(emb_dim, enc_dim=enc_dim, num_head=2, depth=3, dropout=0.1, pre_norm=True,
                                         sn=True, sln=True, w_dim=w_dim, output_atten=True).to(device)
    z, attens_inp = transformer_dec(q, enc_inp, kv, mask_input=mask_inp, mask_source=mask_sou, w=w)

    print(kv.shape, z.shape)
    print(attens_sou[0].shape, attens_inp[0].shape)

    plt.imshow(attens_sou[0][0].detach().cpu().numpy())
    plt.show()