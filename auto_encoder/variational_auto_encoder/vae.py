import torch
import torch.nn.functional as F

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from auto_encoder import AutoEncoder

import os


class VAE(AutoEncoder):
    def __init__(self, feature_size: int, config: dict = None, device: str = 'cpu', lr: float = 0.001, epochs: int = 20) -> None:
        super().__init__(feature_size, config, device, lr, epochs)
        self.config['latent_dim'] = self.config['latent_dim'] * 2   # mean and log_variance
    
    def forward(self, x):
        mean, log_var = torch.chunk(self.encoder(x), 2, dim=1)  # Split the encoded values into mean and log_var components
        z = self.reparameterization(mean=mean, log_var=log_var)
        output = self.decoder(z)

        return output, mean, log_var
    
    def reparameterization(self, mean, log_var):
        epsilon = torch.randn_like(log_var).to(self.device)      
        z = mean + torch.exp(0.5 * log_var) * epsilon   # Use 0.5 * log_var to compute the standard deviation
        return z

    def calc_loss(self, input, output, mean, log_var):
        reconstruction_loss = F.mse_loss(input, output)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reconstruction_loss + kl_divergence
    
    def step(self, batch):
        self.optimizer.zero_grad()
        output, mean, log_var = self.forward(batch)
        loss = self.calc_loss(batch, output, mean, log_var)
        loss.backward()
        self.optimizer.step()
        return loss.item()

if __name__ == '__main__':
    ##### 0. Load Dataset #####
    dataset_name = 'MNIST'
    if dataset_name == 'MNIST':
        dataset = MNIST(root='./data', transform=ToTensor(), download=True)
        channels = 1
        image_size = 28
    elif dataset_name == 'CIFAR-10':
        dataset = CIFAR10(root='./data', transform=ToTensor(), download=True)
        channels = 3
        image_size = 32
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    ##### 1. Train the autoencoder #####
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/{dataset_name}/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    feature_size = channels * image_size * image_size
    vae = VAE(feature_size=feature_size, device=device,
                               config={'type': 'cnn', 
                                       'hidden_sizes': [512, 256, 128, 64, 32],
                                       'channels': channels,
                                       'image_size': image_size,}).to(device)
    vae.learn(dataloader=dataloader, log_dir=log_dir, channels=channels, image_size=image_size)
