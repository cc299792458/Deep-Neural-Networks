"""
    Cycle Generative Adversarial Networks
"""

import os
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid, save_image
from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

from PIL import Image
from IPython.display import HTML
from utils.data_utils import StyleTransferDataset, plot_data_from_dataloader
from utils.misc_utils import set_seed, generate_random_images_and_save
from gan.deep_convolutional_gan import DCGAN, Generator, Discriminator

plt.rcParams['animation.embed_limit'] = 200

class Generator(Generator):
    def __init__(self, config, feature_size=None):
        super().__init__(config, feature_size)

    def forward(self, img):
        return self.model(img)
    
    class ResidualBlock(nn.Module):
        def __init__(self, in_features):
            super().__init__()

            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                nn.InstanceNorm2d(in_features),
            )

        def forward(self, x):
            return x + self.block(x)
        
    def create_generator(self):
        """
            Create generator
            NOTE: Currently hard-coded. Will be parametrized.
        """
        channels = self.config['channels']
        num_residual_block = self.config['num_residual_block']
        
        out_features = 64
        
        # Initial convolution block with corrected padding
        layers = [
            nn.ReflectionPad2d(3),  # Adjusted to match kernel size
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        
        in_features = out_features
        
        # Downsampling
        for _ in range(2):
            out_features *= 2
            layers += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        layers += [self.ResidualBlock(out_features) for _ in range(num_residual_block)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        layers += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        return nn.Sequential(*layers)

class Discriminator(Discriminator):
    def __init__(self, config, feature_size):
        """
            Create discriminator
            NOTE: Currently Hard-coded, too.
        """
        super().__init__(config, feature_size)

    def create_discriminator(self):
        channels = self.config['channels']
        image_size = self.config['image_size']
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, image_size // 2 ** 4, image_size // 2 ** 4)

        layers = [
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)]
            
        return nn.Sequential(*layers)
    
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Buffer size should be a positive number."
        self.max_size = max_size
        self.buffer = []

    def push_and_pop(self, data):
        to_return = []
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(element)
                to_return.append(element)
            else:
                prob = random.random()
                if prob > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.buffer[i].clone())
                    self.buffer[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class CycleGAN(DCGAN):
    def __init__(self, feature_size: int, 
                config: dict = None, 
                device: str = 'cpu', 
                lr: float = 0.0002, 
                betas: tuple[float, float] = (0.5, 0.999), 
                epochs: int = 50,
                decay_epoch: int = 100,
                lambda_id: float = 5.0,
                lambda_cyc: float = 10.0,) -> None:
        
        default_config = {
            'channels': 3,
            'image_size': 256,   # Default for monet2photo
            # 'g_hidden_channels': [512, 256, 128, 64],
            # 'g_kernel_sizes': [4, 4, 4, 4, 1],
            # 'g_strides': [1, 2, 2, 2, 1],
            # 'g_paddings': [0, 1, 1, 1, 2],
            # 'g_batchnorm': True,
            # 'g_activation': nn.ReLU(),
            # 'd_hidden_channels': [64, 128, 256],
            # 'd_kernel_sizes': [4, 4, 4, 4],
            # 'd_strides': [2, 2, 2, 1],
            # 'd_paddings': [1, 1, 2, 0],
            # 'd_batchnorm': True,
            # 'd_activation': nn.LeakyReLU(0.2),
            'num_residual_block': 9,
            'replay_buffer_max_size': 50,

        }

        if config is not None:
            default_config.update(config)    

        super().__init__(feature_size, default_config, device, lr, betas, epochs)

        max_size = self.config.get('replay_buffer_max_size')

        self.decay_epoch = decay_epoch
        self.create_lr_scheduler()
        self.create_replay_buffer(max_size=max_size)

        self.criterion_identity = nn.L1Loss()
        self.criterion_GAN = nn.MSELoss()   # NOTE: how about use BCELoss here?   
        self.criterion_cycle = nn.L1Loss() 
        self.lambda_id = lambda_id
        self.lambda_cyc = lambda_cyc

    def create_networks(self):
        self.generator_A2B = Generator(config=self.config, feature_size=self.feature_size).to(self.device)
        self.generator_B2A = Generator(config=self.config, feature_size=self.feature_size).to(self.device)

        self.discriminator_A = Discriminator(config=self.config, feature_size=self.feature_size).to(self.device)
        self.discriminator_B = Discriminator(config=self.config, feature_size=self.feature_size).to(self.device)

    def create_optimizer(self, lr, betas):
        """
            Create optimizers
        """
        self.g_optimizer = torch.optim.Adam(
            itertools.chain(self.generator_A2B.parameters(), self.generator_B2A.parameters()), lr=lr, betas=betas
        )
        self.d_optimizer_A = torch.optim.Adam(self.discriminator_A.parameters(), lr=lr, betas=betas)
        self.d_optimizer_B = torch.optim.Adam(self.discriminator_B.parameters(), lr=lr, betas=betas)

    def create_lr_lambda(self, n_epochs, decay_start_epoch):
        """
            Create function lr_lambda using for create lr scheduler
        """
        def lr_lambda(epoch):
            return 1.0 - max(0.0, epoch - decay_start_epoch) / (n_epochs - decay_start_epoch)
        return lr_lambda

    def create_lr_scheduler(self):
        """
            Create learing rate schedulers
        """
        lr_lambda = self.create_lr_lambda(self.epochs, self.decay_epoch)

        self.g_lr_scheduler = LambdaLR(self.g_optimizer, lr_lambda=lr_lambda)
        self.d_lr_scheduler_A = LambdaLR(self.d_optimizer_A, lr_lambda=lr_lambda)
        self.d_lr_scheduler_B = LambdaLR(self.d_optimizer_B, lr_lambda=lr_lambda)
    
    def create_replay_buffer(self, max_size):
        self.replay_buffer_A = ReplayBuffer(max_size=max_size)
        self.replay_buffer_B = ReplayBuffer(max_size=max_size)
    
    def learn(self, train_dataloader, test_dataloader, log_dir):
        fixed_train_batch = next(iter(train_dataloader))
        fixed_test_batch = next(iter(test_dataloader))

        fixed_samples_train_A = fixed_train_batch['A'].to(device)
        fixed_samples_train_B = fixed_train_batch['B'].to(device)
        fixed_samples_test_A = fixed_test_batch['A'].to(device)
        fixed_samples_test_B = fixed_test_batch['B'].to(device)

        # Lists to keep track of progress
        img_train_A_list = []
        img_train_B_list = []
        img_test_A_list = []
        img_test_B_list = []
        g_losses = []
        d_A_losses = []
        d_B_losses = []

        print("Starting Training Loop...")
        for epoch in range(self.epochs):
            for step, batch in enumerate(train_dataloader):
                real_A = batch['A'].to(self.device)
                real_B = batch['B'].to(self.device)

                batch_size = real_A.shape[0]
                discriminator_output_shape = self.discriminator_A.output_shape
                # (Patch GAN)
                real_label = torch.ones((batch_size, *discriminator_output_shape), dtype=torch.float32).to(self.device)
                fake_label = torch.zeros((batch_size, *discriminator_output_shape), dtype=torch.float32).to(self.device)

                # Update Generator
                fake_A, fake_B, g_loss = self.generator_step(real_A=real_A, real_B=real_B, real_label=real_label, fake_label=fake_label)

                # Update Discriminator A
                d_A_loss = self.discriminator_A_step(real_A, fake_A, real_label, fake_label)

                # Update Discriminator B
                d_B_loss = self.discriminator_B_step(real_B, fake_B, real_label, fake_label)

                # Update Learning Rate
                self.g_lr_scheduler.step()
                self.d_lr_scheduler_A.step()
                self.d_lr_scheduler_B.step()

                # TODO: Add a print here.

                # TODO: Add evaluation here.

                if epoch % 10 == 0:
                    models_dir = os.path.join(log_dir, 'models')
                    os.makedirs(models_dir, exist_ok=True)

                    model_path = os.path.join(models_dir, f'epoch_{epoch}_model.pth')
                    torch.save(self.state_dict(), model_path)

            models_dir = os.path.join(log_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(models_dir, 'final_model.pth'))


    def generator_step(self, real_A, real_B, real_label):
        self.g_optimizer.zero_grad()

        # Identity loss
        id_loss_A = self.criterion_identity(self.generator_B2A(real_A), real_A)
        id_loss_B = self.criterion_identity(self.generator_A2B(real_B), real_B)

        identity_loss = (id_loss_A + id_loss_B) / 2

        # GAN loss
        fake_B = self.generator_A2B(real_A)
        loss_GAN_AB = self.criterion_GAN(self.discriminator_B(fake_B), real_label)
        fake_A = self.generator_B2A(real_B)
        loss_GAN_BA = self.criterion_GAN(self.discriminator_A(fake_A), real_label)

        GAN_loss = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = self.generator_B2A(fake_B)
        cycle_loss_A = self.criterion_cycle(recov_A, real_A)
        recov_B = self.generator_A2B(fake_A)
        cycle_loss_B = self.criterion_cycle(recov_B, real_B)

        cycle_loss = (cycle_loss_A + cycle_loss_B) / 2

        # Total loss
        g_loss = self.lambda_id * identity_loss + GAN_loss + self.lambda_cyc * cycle_loss

        g_loss.backward()
        self.g_optimizer.step()

        # TODO: return more items (like scores here)
        return fake_A, fake_B, g_loss
    
    def discriminator_A_step(self, real_A, fake_A, real_label, fake_label):
        self.d_optimizer_A.zero_grad()

        # Real loss
        real_loss = self.criterion_GAN(self.discriminator_A(real_A), real_label)
        
        # Fake loss (on batch of previously generated samples)
        reused_fake_A = self.replay_buffer_A.push_and_pop(fake_A)
        fake_loss = self.criterion_GAN(self.discriminator_A(reused_fake_A.detach()), fake_label)
        
        # Total loss
        d_loss_A = (real_loss + fake_loss) / 2

        d_loss_A.backward()
        self.d_optimizer_A.step()

        # TODO: return more items (like scores here)
        return d_loss_A

    def discriminator_B_step(self, real_B, fake_B, real_label, fake_label):
        self.d_optimizer_B.zero_grad()

        # Real loss
        real_loss = self.criterion_GAN(self.discriminator_B(real_B), real_label)
        
        # Fake loss (on batch of previously generated samples)
        reused_fake_B = self.replay_buffer_B.push_and_pop(fake_B)
        fake_loss = self.criterion_GAN(self.discriminator_B(reused_fake_B.detach()), fake_label)
        
        # Total loss
        d_loss_B = (real_loss + fake_loss) / 2

        d_loss_B.backward()
        self.d_optimizer_B.step()

        # TODO: return more items (like scores here)
        return d_loss_B

    def plot_loss_curves(self, g_losses, d_losses_A, d_losses_B, log_dir):
        # TODO: Re-write this function
        pass

    def visualize_progression(self, img_list, dataloader, log_dir):
        # TODO: Re-write this function
        pass


if __name__ == '__main__':
    set_seed()
    ##### 0. Load Dataset #####
    dataset_name = 'monet2photo'
    train_batch_size = 1
    test_batch_size = 5

    if dataset_name == 'monet2photo':
        channels = 3
        image_size = 256
        transform = Compose([Resize(int(256 * 1.12), Image.BICUBIC), 
                             RandomCrop(256),
                             RandomHorizontalFlip(),
                             ToTensor(),
                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        train_dataset = StyleTransferDataset(root='./data/monet2photo', transform=transform, unaligned=True)
        test_dataset = StyleTransferDataset(root='./data/monet2photo', transform=transform, unaligned=True, mode='test')
    feature_size = channels * image_size * image_size
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/{dataset_name}/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # plot_data_from_dataloader(dataloader=train_dataloader, device=device, type='style_transfer')

    ## Training parameters ## 
    epochs = 200
    decay_epoch = 100

    cycle_gan = CycleGAN(feature_size=feature_size, device=device,
            config={'channels': channels, 
                    'image_size': image_size,}, epochs=epochs, decay_epoch=decay_epoch).to(device)

    train = True
    ##### 1. Train the model #####
    if train:
        cycle_gan.learn(dataloader=train_dataloader, log_dir=log_dir)

