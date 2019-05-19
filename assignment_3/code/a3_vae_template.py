import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from datetime import datetime
from torchvision.utils import make_grid

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        # a gaussian encoder
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = torch.tanh(self.hidden(input))
        return self.mean(h), self.std(h)


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

        mean, std = self.encoder.forward(input)
        epsilon = torch.zeros(mean.shape).normal_() #check dimensions!

        z = std * epsilon + mean
        y = self.decoder.forward(z)

        l_reconstruction = - (input * torch.log(y) + (1-input) * (1-torch.log(y))) # check dimensions! check sign! dot product?

        l_regularize = torch.log(std) + 0.5 * (std**2 + mean**2) - 0.5


        average_negative_elbo = torch.mean(torch.sum(l_reconstruction) + torch.sum(l_regularize))
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims, im_means = None, None
        raise NotImplementedError()

        return sampled_ims, im_means


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
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=10)  # prevents maximum gradient problem
            optimizer.step()
        average_epoch_elbo.append(elbo)

        if step % 10 == 0:
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
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--h_dim', default=500, type=int,
                        help='dimensionality of hidden space')

    ARGS = parser.parse_args()

    main()
