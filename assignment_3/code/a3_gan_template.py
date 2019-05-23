import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
from statistics import mean as mean_
import math


class Generator(nn.Module):
    def __init__(self, latent_dim, input_dim=784, h_dim=128, device=None):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        # in the generator we go from latent space  z to input space x
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, h_dim),  # latent_dim -> 128
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, 2 * h_dim),  # 128 -> 256
            nn.BatchNorm1d(2 * h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * h_dim, 4 * h_dim),  # 256 -> 512
            nn.BatchNorm1d(4 * h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(4 * h_dim, 8 * h_dim),  # 512 -> 1024
            nn.BatchNorm1d(8 * h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(8 * h_dim, input_dim),  # 1024 -> input_dim
            nn.Tanh()
        )
        self.to(device)

    def forward(self, z):
        return self.generator(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=784, h_dim=512, device=None):
        super(Discriminator, self).__init__()
        self.device = device
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, h_dim),  # input_dim -> 512
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(h_dim, int(h_dim / 2)),  # 512 -> 256
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(int(h_dim / 2), 1),  # 256 -> 1
            nn.Sigmoid()
        )
        self.to(device)

    def forward(self, img):
        return self.discriminator(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):

    gen_curve, disc_curve = [], []
    for epoch in range(args.n_epochs):
        losses_d, losses_g = [], []
        for step, (imgs, _) in enumerate(dataloader):
            imgs = imgs.reshape(imgs.shape[0], -1).to(device) # [batch_size, 784] make images 1-dimensional


            D_X = discriminator.forward(imgs)

            # Sample
            z = torch.randn((imgs.shape[0], generator.latent_dim)).to(device)
            generated_img = generator.forward(z)
            D_GZ = discriminator.forward(generated_img)
            loss_g = -torch.mean(torch.log(D_GZ))

            # New sample
            z = torch.randn((imgs.shape[0], generator.latent_dim)).to(device)
            generated_img = generator.forward(z)
            D_GZ = discriminator.forward(generated_img)
            loss_d = -(torch.mean(torch.log(D_X)) + torch.mean(torch.log(1-D_GZ)))

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            # Train Generator
            # ---------------
            loss_g.backward(retain_graph=True)
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            loss_d.backward()
            optimizer_D.step()

            losses_d.append(loss_d.item())
            losses_g.append(loss_g.item())

            # Save Images
            # -----------

            batches_done = epoch * len(dataloader) + step
            if batches_done % args.save_interval == 0:
                # Sample
                z = torch.randn((args.n_samples, generator.latent_dim)).to(device)
                generated_img = generator.forward(z)
                generated_img = generated_img.reshape(-1, 1, 28, 28)
                save_image(generated_img, f"gan_images/grid_Epoch{epoch}.png", nrow=int(math.sqrt(args.n_samples)), padding=2, normalize=True)

            # if step % args.print_every == 0:
            #     print("[{}] Loss_G = {}, Loss_D = {} ".format(datetime.now().strftime("%Y-%m-%d %H:%M"), loss_g.item(), loss_d.item()) + '\n')

        print(f"[Epoch {epoch}] loss generator: {losses_g[-1]} loss discriminator: {losses_d[-1]}")

        gen_curve.append(mean_(losses_g))
        disc_curve.append(mean_(losses_d))
        save_elbo_plot(gen_curve, disc_curve, 'gan_images/loss.pdf')
    save_elbo_plot(gen_curve, disc_curve, 'gan_images/loss.pdf')


def save_elbo_plot(generator_curve, discriminator_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(generator_curve, label='generator')
    plt.plot(discriminator_curve, label='discriminator')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    # Create output image directory
    os.makedirs('gan_images', exist_ok=True)
    device = torch.device(args.device)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim, device=device)
    discriminator = Discriminator(device=device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device)

    torch.save(generator.state_dict(), args.save_model)


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
    parser.add_argument('--save_model', type=str, default="./gan_generator.pt",
                        help="Path to a file to save the model on")
    parser.add_argument('--n_samples', default=16, type=int,
                        help='number of samples')
    parser.add_argument('--print_every', default=1000, type=int,
                        help='print every step')
    args = parser.parse_args()

    main()
