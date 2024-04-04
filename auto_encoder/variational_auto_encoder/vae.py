import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from auto_encoder import AutoEncoder
from utils.misc_utils import set_seed


class VAE(AutoEncoder):
    def __init__(self, feature_size: int, config: dict = None, device: str = 'cpu', lr: float = 1e-3, epochs: int = 30) -> None:
        super().__init__(feature_size, config, device, lr, epochs)
        self.double_output_dim_of_encoder_last_layer()
        
    
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
        reconstruction_loss = F.binary_cross_entropy(output, input, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reconstruction_loss + kl_divergence
    
    def step(self, batch):
        self.optimizer.zero_grad()
        output, mean, log_var = self.forward(batch)
        loss = self.calc_loss(batch, output, mean, log_var)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def double_output_dim_of_encoder_last_layer(self):
        # Verify the final layer is a Linear layer
        if not isinstance(self.encoder[-1], nn.Linear):
            raise ValueError("The final layer of the encoder must be an nn.Linear layer.")

        last_layer = self.encoder[-1]
        in_features, out_features = last_layer.in_features, last_layer.out_features
        self.encoder[-1] = nn.Linear(in_features, out_features * 2)

    def reconstruct_and_save_image(self, sample, log_dir, channels, image_size, epoch=None):
        epoch = 'init' if epoch is None else f'epoch_{epoch}'
        with torch.no_grad():
            reconstructed_sample = self.forward(sample)[0].reshape(channels, image_size, image_size)
            reconstructed_image = to_pil_image(reconstructed_sample.cpu())
            save_path = os.path.join(log_dir, f'reconstructed_image_{epoch}.png')
            reconstructed_image.save(save_path)

if __name__ == '__main__':
    set_seed()
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
    feature_size = channels * image_size * image_size
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/{dataset_name}/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ##### 1. Train the autoencoder #####    
    # vae = VAE(feature_size=feature_size, device=device,
    #                            config={'type': 'fc', 
    #                                    'latent_dim': 2,
    #                                    'hidden_sizes': [512, 512],
    #                                    'channels': channels,
    #                                    'image_size': image_size,}).to(device)
    # vae.learn(dataloader=dataloader, log_dir=log_dir, channels=channels, image_size=image_size)

    ##### 2. Generate image from random noise #####
    ## Load Model ##
    vae = VAE(feature_size=feature_size, device=device,
                               config={'type': 'fc', 
                                       'latent_dim': 2,
                                       'hidden_sizes': [512, 512],
                                       'channels': channels,
                                       'image_size': image_size,}).to(device=device)
    vae.load_state_dict(torch.load(os.path.join(log_dir, f'best_model.pth')))

    ## Sample ##
    sample_range = [-1, 1]
    steps = 10
    points = torch.linspace(sample_range[0], sample_range[1], steps)
    x, y = torch.meshgrid(points, points)
    z = torch.stack([x.flatten(), y.flatten()], dim=1).to(device)

    samples = [vae.sample(z[i].unsqueeze(0)).reshape(channels, image_size, image_size) for i in range(steps**2)]
    images = [to_pil_image(sample.cpu()) for sample in samples]

    ## Plot ##
    fig, axs = plt.subplots(steps, steps, figsize=(10, 10))
    for ax, img, xi, yi in zip(axs.flatten(), images, x.flatten(), y.flatten()):
        ax.imshow(img)
        ax.axis('off')
        # ax.text(0.5, -0.1, f'({xi:.2f}, {yi:.2f})', va='center', ha='center', fontsize=6, transform=ax.transAxes)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.suptitle('VAE: Random Samples')
    plt.show()