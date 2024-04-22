"""
    DCGAN (Deep Convolutional Generative Adversarial Networks)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

from utils.misc_utils import set_seed, plot_data_from_dataloader, generate_random_images_and_save

from gan import GAN, Generator, Discriminator

plt.rcParams['animation.embed_limit'] = 100

class Generator(Generator):
    def __init__(self, config, feature_size=None):
        super().__init__(config, feature_size)

    def forward(self, z):
        
        return self.model(z)

    def create_generator(self):
        layers = []
        # init_size = self.image_size // 4   # 28 // 4 = 7
        channels = self.config['channels']
        latent_dim = self.config['latent_dim']
        hidden_channels = self.config['g_hidden_channels']
        kernel_sizes = self.config['g_kernel_sizes']
        strides = self.config['g_strides']
        paddings = self.config['g_paddings']
        batchnorm = self.config['g_batchnorm']
        activation = self.config['g_activation']

        # layers.append(nn.Linear(self.latent_dim, hidden_channels[0] * init_size * init_size))
        # layers.append(nn.Unflatten(1, (hidden_channels[0], init_size, init_size)))
        # layers.append(nn.BatchNorm2d(hidden_channels[0]))
        # layers.append(self.g_activation)
        layers.append(nn.ConvTranspose2d(latent_dim, hidden_channels[0],
                                             kernel_size=kernel_sizes[0], stride=strides[0],
                                             padding=paddings[0]))
        if batchnorm:
            layers.append(nn.BatchNorm2d(num_features=hidden_channels[0]))
        layers.append(activation)
        for i in range(len(hidden_channels) - 1):
            layers.append(nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i+1],
                                             kernel_size=kernel_sizes[i+1], stride=strides[i+1],
                                             padding=paddings[i+1], bias=False))
            if batchnorm:
                layers.append(nn.BatchNorm2d(num_features=hidden_channels[i+1]))
            layers.append(activation)
        layers.append(nn.ConvTranspose2d(in_channels=hidden_channels[-1], out_channels=channels,
                                         kernel_size=kernel_sizes[-1], stride=strides[-1], 
                                         padding=paddings[-1]))
        # layers.append(nn.BatchNorm2d(num_features=self.channels))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(Discriminator):
    def __init__(self, config, feature_size):
        super().__init__(config, feature_size)
    
    def forward(self, img):

        return self.model(img)

    def create_discriminator(self):
        layers = []
        hidden_channels = self.config['d_hidden_channels']
        kernel_sizes = self.config['d_kernel_sizes']
        strides = self.config['d_strides']
        paddings = self.config['d_paddings']
        in_channels = self.config['channels']
        # image_size = self.image_size
        batchnorm = self.config['d_batchnorm']
        activation = self.config['d_activation']

        for i in range(len(hidden_channels)):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels[i],
                                    kernel_size=kernel_sizes[i], stride=strides[i],
                                    padding=paddings[i], bias=False))
            if batchnorm:
                layers.append(nn.BatchNorm2d(num_features=hidden_channels[i]))
            layers.append(activation)
            # layers.append(nn.Dropout2d(0.25))
            in_channels = hidden_channels[i]
            # image_size = (image_size - kernel_sizes[i] + 2 * paddings[i]) // strides[i] + 1
        # final_num_features = hidden_channels[-1] * image_size * image_size
        # layers.append(nn.Flatten())
        # layers.append(nn.Linear(in_features=final_num_features, out_features=1))
        layers.append(nn.Conv2d(hidden_channels[-1], 1, 
                                kernel_size=kernel_sizes[-1], stride=strides[-1], 
                                padding=paddings[-1], bias=False))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

class DCGAN(GAN):
    def __init__(self, 
                feature_size: int, 
                config: dict = None, 
                device: str = 'cpu', 
                lr: float = 0.0002, 
                betas: tuple[float, float] = (0.5, 0.999), 
                epochs: int = 50) -> None:
        
        default_config = {
            'channels': 1,
            'image_size': 28,   # Default for MNIST
            'latent_dim': 128,
            ## For Convolutional Network ##
            'generator_cls': Generator,
            'g_hidden_channels': [512, 256, 128, 64],
            'g_kernel_sizes': [4, 4, 4, 4, 1],
            'g_strides': [1, 2, 2, 2, 1],
            'g_paddings': [0, 1, 1, 1, 2],
            'g_batchnorm': True,
            'g_activation': nn.ReLU(),
            'discriminator_cls': Discriminator, 
            'd_hidden_channels': [64, 128, 256],
            'd_kernel_sizes': [4, 4, 4, 4],
            'd_strides': [2, 2, 2, 1],
            'd_paddings': [1, 1, 2, 0],
            'd_batchnorm': True,
            'd_activation': nn.LeakyReLU(0.2),
        }

        if config is not None:
            default_config.update(config)

        super().__init__(feature_size, default_config, device, lr, betas, epochs)
    
    def sample_z(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)

        return z
    
    def sample(self, z=None, batch_size=None):
        z = self.sample_z(batch_size=batch_size) if z == None else z
        while len(z.shape) < 4:
            z = z.unsqueeze(-1)
        output = self.generator(z)

        return output

if __name__ == '__main__':
    set_seed()
    ##### 0. Load Dataset #####
    dataset_name = 'MNIST'
    batch_size = 128

    if dataset_name == 'MNIST':
        channels = 1
        image_size = 28
        transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
        dataset = MNIST(root='./data', transform=transform, download=True)
    elif dataset_name == 'CIFAR-10':
        channels = 3
        image_size = 32
        transform = Compose([ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        dataset = CIFAR10(root='./data', transform=transform, download=True)
    feature_size = channels * image_size * image_size
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/{dataset_name}/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # plot_data_from_dataloader(dataloader=dataloader, device=device)

    ## Training parameters ## 
    latent_dim = 128
    epochs = 100

    dcgan = DCGAN(feature_size=feature_size, device=device,
            config={'latent_dim': latent_dim, 
                    'channels': channels, 
                    'image_size': image_size,}, epochs=epochs).to(device)

    train = True
    ##### 1. Train the model #####
    if train:
        dcgan.learn(dataloader=dataloader, log_dir=log_dir)

    ##### 2. Generate image from random noise #####
    else:
        ## Load Model ##
        model_path = os.path.join(log_dir, 'models/final_model.pth')
        dcgan.load_state_dict(torch.load(model_path))

        num_images = 400
        z_ranges = ((-1, 1), (-1, 1))
        generate_random_images_and_save(dcgan, 
                                        num_images=num_images, 
                                        log_dir=log_dir, 
                                        image_size=image_size, 
                                        latent_dim=latent_dim)