import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils.misc_utils import set_seed, generate_random_images_and_save, generate_uniformly_distributed_images_and_save


class GAN(nn.Module):
    def __init__(self,
                config: dict = None,
                device: str = 'cpu', 
                lr: float = 2e-4, 
                betas: tuple[float, float] = (0.5, 0.999),
                epochs: int = 200) -> None:
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

    def sample(self):
        pass

if __name__ == '__main__':
    set_seed()
    ##### 0. Load Dataset #####
    dataset_name = 'MNIST'
    batch_size = 64
    if dataset_name == 'MNIST':
        dataset = MNIST(root='./data', transform=ToTensor(), download=True)
        channels = 1
        image_size = 28
    elif dataset_name == 'CIFAR-10':
        dataset = CIFAR10(root='./data', transform=ToTensor(), download=True)
        channels = 3
        image_size = 32
    feature_size = channels * image_size * image_size
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/{dataset_name}/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ## Training parameters ## 
    model_type = 'cnn'
    latent_dim = 2

    gan = GAN(feature_size=feature_size, device=device, 
              config={'type': model_type, 
                      'latent_dim': latent_dim, 
                      'channels': channels, 
                      'image_size': image_size,}).to(device)
    
    train = True
    # ##### 1. Train the autoencoder #####
    if train:
        gan.learn(dataloader=dataloader, log_dir=log_dir, channels=channels, image_size=image_size)
    
    ##### 2. Generate image from random noise #####
    else:
        ## Load Model ##
        gan.load_state_dict(torch.load(os.path.join(log_dir, f'best_model.pth')))

        num_images = 400
        z_ranges = ((-1, 1), (-1, 1))
        generate_random_images_and_save(gan, 
                                        num_images=num_images, 
                                        log_dir=log_dir, 
                                        image_size=image_size, 
                                        latent_dim=latent_dim)
        generate_uniformly_distributed_images_and_save(gan, 
                                                    num_images=num_images, 
                                                    z_ranges=z_ranges, 
                                                    log_dir=log_dir, 
                                                    image_size=image_size, 
                                                    latent_dim=latent_dim)