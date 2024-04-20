"""
    Conditional Generative Adversarial Networks (CGAN)
"""

import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision.transforms import Compose, ToTensor, Normalize

from utils.misc_utils import set_seed, plot_data_from_dataloader, generate_random_images_and_save

from gan.deep_convolutional_gan import DCGAN

class CGAN(DCGAN):
    """
        Conditional Generative Adversarial Networks
    """

    def __init__(self, 
                feature_size: int, 
                config: dict = None, 
                device: str = 'cpu', lr: float = 0.0002, 
                betas: tuple[float, float] = (0.5, 0.999), 
                epochs: int = 50) -> None:
        default_config = {
            'num_classes': 10,
        }
        
        if config is not None:
            default_config.update(config)
        
        super().__init__(feature_size, config, device, lr, betas, epochs)

        self.num_classes = self.config.get('num_classes')
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)

        self.modify_generator()
        self.modify_discriminator()
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

    def modify_generator(self):
        first_layer = list(self.generator.children())[0]
        modified_first_layer = nn.ConvTranspose2d(self.latent_dim, first_layer.out_channels,
                                                  kernel_size=first_layer.kernel_size, 
                                                  stride=first_layer.stride,
                                                  padding=first_layer.padding)
        new_layers = [modified_first_layer] + list(self.generator.children())[1:]
        self.generator = nn.Sequential(*new_layers)

    def modify_discriminator(self):
        first_layer = list(self.discriminator.children())[0]
        modified_first_layer = nn.Conv2d(first_layer.in_channels + self.num_classes, first_layer.out_channels,
                                         kernel_size=first_layer.kernel_size, 
                                         stride=first_layer.stride,
                                         padding=first_layer.padding)
        new_layers = [modified_first_layer] + list(self.discriminator.children())[1:]
        self.discriminator = nn.Sequential(*new_layers)

    def learn(self):
        # TODO: Add training process here.
        pass

    def sample_z(self, batch_size, labels):
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        labels = self.label_emb(labels).view(batch_size, self.num_classes, 1, 1)
        combined_input = torch.cat([z, labels], 1)
        return combined_input