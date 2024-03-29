import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.stats as stats


import time
from datetime import datetime
import numpy as np
from torchvision.utils import save_image
import math

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.z_dim = z_dim
        # a gaussian encoder
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.log_std = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = torch.tanh(self.hidden(input))
        mean, log_std = self.mean(h), self.log_std(h)
        return mean, log_std


class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        # a bernoulli decoder
        self.decode = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.decode(input)
        return mean


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(input_dim, hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        mean, log_var = self.encoder.forward(input)
        var = 2**log_var
        epsilon = torch.zeros(mean.shape).normal_() # we cannot use z_dim because we're using batches

        z = torch.sqrt(var) * epsilon + mean
        y = self.decoder.forward(z)

        l_reconstruction = - (input * torch.log(y) + (1-input) * torch.log(1-y) )

        l_regularize = 0.5 * (-log_var + var + mean**2 - 1)
        # l_regularize = 0.5 * (torch.log(std**2) + std ** 2 + mean ** 2 - 1)

        average_negative_elbo = torch.mean(torch.sum(l_reconstruction, dim=1) + torch.sum(l_regularize, dim=1))
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        samples = torch.zeros(n_samples, self.z_dim).normal_()

        # Careful! Do not update the weights
        with torch.no_grad():
            img_means = self.decoder(samples)
        sampled_imgs = (torch.rand(img_means.shape) < img_means)
        return sampled_imgs, img_means

    def get_manifold(self, i):
        epsilon = 1e-3
        vals = np.linspace(0 + epsilon, 1 - epsilon, i)
        z = torch.stack([torch.tensor(stats.norm.ppf([x, y])).float() for x in vals for y in vals])

        with torch.no_grad():
            mean = self.decoder.forward(z)

        # xy = np.mgrid[0:i, 0:i].reshape((2, i ** 2)).T / (i - 1)
        # xy = (xy + 4.45e-2) * 9e-1
        #
        # z = torch.tensor(stats.norm.ppf(xy), dtype=torch.float)
        # with torch.no_grad():
        #     mean = self.decoder.forward(z)
        return mean


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = []

    for step, batch in enumerate(data):
        t1 = time.time()

        batch = batch.reshape(batch.shape[0], -1) #batch = [128, 784=28*28] column known, row unknown
        elbo = model.forward(batch)

        if model.training:
            model.zero_grad()
            elbo.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=10)  # prevents maximum gradient problem
            optimizer.step()
        average_epoch_elbo.append(elbo.item())

        if step % ARGS.print_every == 0:
            print("[{}] Loss = {} ".format(datetime.now().strftime("%Y-%m-%d %H:%M"), elbo) + '\n')
        t2 = time.time()
    return np.mean(average_epoch_elbo)


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    # 28 is image dimension
    model = VAE(28*28, hidden_dim=ARGS.h_dim, z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        # --------------------------------------------------------------------

        imgs, means = model.sample(n_samples=ARGS.n_samples)

        # Only use means for better visualization
        # Reshape the data to [n_samples, 1, 28, 28]
        means = means.reshape(-1, 1, 28, 28)
        save_image(means, f"grid_Epoch{epoch}.png", nrow=int(math.sqrt(ARGS.n_samples)), padding=2, normalize=True)


    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

        if ARGS.zdim == 2:
            manifold_means = model.get_manifold(20)
            manifold_means = manifold_means.reshape(-1, 1, 28, 28)
            save_image(manifold_means, f"manifold{epoch}.png", nrow=20, padding=2, normalize=True)

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--h_dim', default=500, type=int,
                        help='dimensionality of hidden space')
    parser.add_argument('--n_samples', default=16, type=int,
                        help='number of samples')
    parser.add_argument('--print_every', default=100, type=int,
                        help='print every step')
    ARGS = parser.parse_args()

    main()
