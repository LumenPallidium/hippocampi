import torch
from tqdm import tqdm

class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size = 3, 
                 stride = 1,
                 padding = 1,
                 activation = torch.nn.GELU()):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.conv_skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv_skip = torch.nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.activation(y)
        y = self.conv2(y)
        y = self.bn2(y)
        res = self.conv_skip(x)
        return self.activation(y + res)
    
class RescaleBlock(torch.nn.Module):
    """
    Simple rescaling + resnet block.
    """
    def __init__(self,
                 scale,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 mode = "bilinear"):
        super().__init__()
        self.scale = scale
        self.resnet = ResnetBlock(in_channels, out_channels, kernel_size, stride, padding)

        self.mode = mode

    def forward(self, x):
        x_rescale = torch.nn.functional.interpolate(x,
                                                    scale_factor=self.scale,
                                                    mode=self.mode)
        return self.resnet(x_rescale)
    
class MNISTAutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim = 128):
        super().__init__()
        self.encoder_conv = torch.nn.Sequential(
            RescaleBlock(0.5, 1, 16),
            RescaleBlock(0.5, 16, 32)
        )
        self.encoder_linear = torch.nn.Linear(32 * 7 * 7, latent_dim)
        self.tanh = torch.nn.Tanh()
        self.decoder_conv = torch.nn.Sequential(
            RescaleBlock(2, 32, 16),
            RescaleBlock(2, 16, 1)
        )
        self.decoder_linear = torch.nn.Linear(latent_dim, 32 * 7 * 7)

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, 32 * 7 * 7)
        z = self.encoder_linear(x)
        return self.tanh(z)

    def decode(self, z):
        x_hat = self.decoder_linear(z)
        x_hat = x_hat.view(-1, 32, 7, 7)
        x_hat = self.decoder_conv(x_hat)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat
    
def mnist_sae_train(mnist_data, model,
                    plot_recons = False, 
                    n_rows = 4, n_cols = 4,
                    batch_size = 256,
                    epochs = 5, sparsity_weight = 1):
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    pbar = tqdm(range(epochs * len(mnist_data) // batch_size))
    for epoch in range(epochs):
        dl = torch.utils.data.DataLoader(mnist_data,
                                        batch_size = batch_size,
                                        shuffle = True)
        for x, _ in dl:
            x = x.to(device)
            optimizer.zero_grad()
            z, x_hat = model(x)
            loss = (x - x_hat).pow(2).mean() + sparsity_weight * z.abs().mean()
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description(f"{epoch} Loss: {loss.item():.2f}")
    
    if plot_recons:
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid
        im_pair = torch.concat([x,
                                x_hat],
                                dim = 3)
        im_pair = im_pair[:n_rows * n_cols]
        display_ims = make_grid(im_pair, nrow = n_cols)
        plt.imshow(display_ims[0, :, :].detach().cpu().numpy())
        plt.show()

    return model, z

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])

    mnist = datasets.MNIST("data/", train = True,
                           download = True, transform = transform)
    model = MNISTAutoEncoder()
    model, z = mnist_sae_train(mnist, model,
                    epochs = 5,
                    plot_recons=True)
    print("Training done!")

    # histogram z values, should be exponential given L1
    plt.hist(z.detach().cpu().numpy().flatten(), bins = 100)
    plt.show()
