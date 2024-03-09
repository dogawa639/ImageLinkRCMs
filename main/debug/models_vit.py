if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from models.general import *
    from models.vit import *

    import matplotlib.pyplot as plt

    device = "cpu"

    img_size = (256, 256, 3)
    patch_size = (16, 16)
    num_classes = 10
    dim = 256
    depth = 1
    heads = 1
    output_atten = True

    x = torch.randn(1, 3, 256, 256).to(device)
    model = ViT(img_size, patch_size, num_classes, dim, depth, heads, output_atten=output_atten).to(device)
    y, atten = model(x)  # expected to be shape(1, num_classes)
    print(y.shape)
    print(y.min(), y.max())

    atten = atten[0].reshape(model.split_height, model.split_width).detach().cpu().numpy()

    plt.imshow(atten)
    plt.show()

