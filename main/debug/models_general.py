if __name__ == "__main__":
    import torch
    from models.general import *

    b = 2
    in_c = 3
    out_c = 4
    w_dim = 5
    device = "mps"

    ff = FF(in_c, out_c, hidden_channel=10, act_fn=lambda x : -softplus(x), bias=True, sn=True).to(device)
    sln = SLN(w_dim, in_c).to(device)

    w = torch.randn(b, w_dim).to(device)
    inputs = torch.randn(b, in_c).to(device)

    sln.set_w(w)
    hidden = sln(inputs)
    out = ff(hidden)

    print(inputs[0].detach().cpu().numpy())
    print(hidden.detach().cpu().numpy())
    print(out.shape)

    sln.train()
    w = w[0]
    sln.set_w(w)
    hidden = sln(inputs)
    out = ff(hidden)
    out2 = torch.randn_like(out)
    loss = torch.norm(torch.max(out - out2, torch.tensor(-50.0, device=device)))
    loss.backward()

    print(inputs[0].detach().cpu().numpy())
    print(hidden.detach().cpu().numpy())
    print(out.shape)

    for param in sln.parameters():
        print(param.grad)

