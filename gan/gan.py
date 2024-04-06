import os
import torch
import torch.nn as nn
import torch.optim as optim


class GAN(nn.Module):
    def __init__(self,
                config: dict = None,
                device: str = 'cpu', 
                lr: float = 1e-3, 
                epochs: int = 100) -> None:
        super(GAN, self).__init__()
        self.device = device
        self.config = {
            'type': 'cnn',
            'channels': 1,
            'latent_dim': 2,
            'image_size': 28,   # Default for MNIST
            'generator_hidden_channels': [64, 128],    # Used for CNN #
            'discriminator_hidden_channels': [128, 64],    # Used for CNN #
            'kernel_sizes': [4, 4],
            'strides': [2, 2],
            'paddings': [1, 1],
            'activation': nn.ReLU()
        }

        if config is not None:
            self.config.update(config)

        self.architecture_type = self.config.get('type')
        self.activation = self.config.get('activation')

        if self.architecture_type == 'cnn':
            self.generator = self.create_cnn_generator()
            self.discriminator = self.create_cnn_discriminator()
        else:
            raise ValueError("Unsupported architecture type")

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.epochs = epochs

    def create_cnn_generator(self):
        pass

    def create_cnn_discriminator(self):
        pass

    def step(self):
        pass
    
    def learn(self, dataloader, log_dir, channels, image_size, patience=8, delta=0):
        pass
