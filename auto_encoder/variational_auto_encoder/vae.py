import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from auto_encoder import AutoEncoder

import os


class VAE(AutoEncoder):
    def __init__(self, feature_size: int, config: dict = None, device: str = 'cpu', lr: float = 0.001, epochs: int = 20) -> None:
        # self.config['latent_dim'] = self.config['latent_dim'] * 2   # mean and log_variance
        super().__init__(feature_size, config, device, lr, epochs)
        
    
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
    
    def create_fc_encoder(self):
        layers = []
        input_size = self.feature_size
        for hidden_size in self.config['hidden_sizes']:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.config['latent_dim'] * 2))     # mean and log_variance
        return nn.Sequential(*layers)
    
    def create_cnn_encoder(self):
        input_channels = self.config['channels']
        image_size = self.config['image_size']
        encoder_hidden_channels = self.config['encoder_hidden_channels']
        encoder_kernel_sizes = self.config['encoder_kernel_sizes']
        encoder_strides = self.config['encoder_strides']
        encoder_paddings = self.config['encoder_paddings']
        latent_dim = self.config['latent_dim'] * 2  # mean and log_variance

        layers = []
        in_channels = input_channels
        for out_channels, kernel_size, stride, padding in zip(encoder_hidden_channels, encoder_kernel_sizes, encoder_strides, encoder_paddings):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU()
            ]
            in_channels = out_channels
            image_size = (image_size - kernel_size + 2 * padding) // stride + 1

        layers += [
            nn.Flatten(),
            nn.Linear(in_channels * image_size * image_size, latent_dim)
        ]

        self.reduced_size = image_size

        return nn.Sequential(*layers)
    
    def reconstruct_and_save_image(self, sample, log_dir, channels, image_size, epoch=None):
        epoch = 'init' if epoch is None else f'epoch_{epoch}'
        with torch.no_grad():
            reconstructed_sample = self.forward(sample)[0].reshape(channels, image_size, image_size)
            reconstructed_image = to_pil_image(reconstructed_sample.cpu())
            save_path = os.path.join(log_dir, f'reconstructed_image_{epoch}.png')
            reconstructed_image.save(save_path)

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
                               config={'type': 'fc', 
                                       'hidden_sizes': [512, 256, 128, 64, 32],
                                       'channels': channels,
                                       'image_size': image_size,}).to(device)
    vae.learn(dataloader=dataloader, log_dir=log_dir, channels=channels, image_size=image_size)
