"""
    Wasserstein Generative Adversarial Networks - Gradient Penalty
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

from ..wgan import WGAN

class WGAN_GP(WGAN):
    def __init__(self, 
            feature_size: int, 
            config: dict = None, 
            device: str = 'cpu', 
            lr: float = 0.0002, 
            betas: tuple[float, float] = (0.5, 0.999), 
            epochs: int = 50) -> None:
        super().__init__(feature_size, config, device, lr, betas, epochs)

        # default_config =

        # if config is not None:
        #     default_config.update(config)

        # super().__init__(feature_size, default_config, device, lr, betas, epochs)

        