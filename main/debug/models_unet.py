if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from models.general import *
    from models.unet import *

    device = "mps"

    x = torch.randn(1, 3, 256, 256).to(device)
    model = UNet(3, 1).to(device)
    y = model(x)
    print(y.shape)
    print(y.min(), y.max())

    plt.imshow(x[0].detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()

    plt.imshow(y[0][0].detach().cpu().numpy())
    plt.show()