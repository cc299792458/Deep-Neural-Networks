"""
    Cycle Generative Adversarial Networks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision.transforms import Compose, ToTensor, Normalize

from IPython.display import HTML
from utils.misc_utils import set_seed, plot_data_from_dataloader, generate_random_images_and_save
from gan.deep_convolutional_gan import DCGAN, Generator, Discriminator

plt.rcParams['animation.embed_limit'] = 200

class Generator(Generator):
    def __init__(self, config, feature_size=None):
        super().__init__(config, feature_size)

    def create_generator(self):
        return super().create_generator()

    def forward(self, z):
        return super().forward(z)


class Discriminator(Discriminator):
    def __init__(self, config, feature_size):
        super().__init__(config, feature_size)

    def forward(self, img):
        return super().forward(img)

    def create_discriminator(self):
        return super().create_discriminator()
    


class CycleGAN(DCGAN):
    def __init__(self, feature_size: int, 
                config: dict = None, 
                device: str = 'cpu', 
                lr: float = 0.0002, 
                betas: tuple[float, float] = (0.5, 0.999), 
                epochs: int = 50) -> None:
        super().__init__(feature_size, config, device, lr, betas, epochs)

if __name__ == '__main__':
    set_seed()
    ##### 0. Load Dataset #####
    dataset_name = 'monet2photo'
    batch_size = 1

