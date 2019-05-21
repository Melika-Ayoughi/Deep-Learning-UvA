import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self, latent_dim, input_dim=784, h_dim=128):
        super(Generator, self).__init__()
        # in the generator we go from latent space  z to input space x

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, h_dim),  # latent_dim -> 128
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, 2 * h_dim),  # 128 -> 256
            nn.BatchNorm1d(),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * h_dim, 4 * h_dim),  # 256 -> 512
            nn.BatchNorm1d(),
            nn.LeakyReLU(0.2),
            nn.Linear(4 * h_dim, 8 * h_dim),  # 512 -> 1024
            nn.BatchNorm1d(),
            nn.LeakyReLU(0.2),
            nn.Linear(8 * h_dim, input_dim),  # 1024 -> input_dim
            nn.Tanh()
        )

    def forward(self, z):
        return self.generator(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=784, h_dim=512):
        super(Discriminator, self).__init__()

        self.descriminator = nn.Sequential(
            nn.Linear(input_dim, h_dim),  # input_dim -> 512
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, h_dim / 2),  # 512 -> 256
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim / 2, 1),  # 256 -> 1
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.descriminator(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            imgs.cuda()

            # Train Generator
            # ---------------
            with optimizer_D.no_grad:

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                pass


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    args = parser.parse_args()

    main()
