"""
    DCGAN (Deep Convolutional Generative Adversarial Networks)
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
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

from IPython.display import HTML
from utils.misc_utils import set_seed, plot_data_from_dataloader, generate_random_images_and_save

from gan import GAN

plt.rcParams['animation.embed_limit'] = 40

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
            'g_hidden_channels': [512, 256, 128, 64],
            'g_kernel_sizes': [4, 4, 4, 4, 1],
            'g_strides': [1, 2, 2, 2, 1],
            'g_paddings': [0, 1, 1, 1, 2],
            'g_activation': nn.ReLU(),
            'd_hidden_channels': [64, 128, 256],
            'd_kernel_sizes': [4, 4, 4],
            'd_strides': [2, 2, 2, 1],
            'd_paddings': [1, 1, 2, 0],
            'd_activation': nn.LeakyReLU(0.2),
        }

        if config is not None:
            default_config.update(config)

        super().__init__(feature_size, default_config, device, lr, betas, epochs)
        
    def create_generator(self):
        layers = []
        # init_size = self.image_size // 4   # 28 // 4 = 7
        hidden_channels = self.config['g_hidden_channels']
        kernel_sizes = self.config['g_kernel_sizes']
        strides = self.config['g_strides']
        paddings = self.config['g_paddings']

        # layers.append(nn.Linear(self.latent_dim, hidden_channels[0] * init_size * init_size))
        # layers.append(nn.Unflatten(1, (hidden_channels[0], init_size, init_size)))
        # layers.append(nn.BatchNorm2d(hidden_channels[0]))
        # layers.append(self.g_activation)
        layers.append(nn.ConvTranspose2d(self.latent_dim, hidden_channels[0],
                                             kernel_size=kernel_sizes[0], stride=strides[0],
                                             padding=paddings[0]))
        layers.append(nn.BatchNorm2d(num_features=hidden_channels[0]))
        layers.append(self.g_activation)
        for i in range(len(hidden_channels) - 1):
            layers.append(nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i+1],
                                             kernel_size=kernel_sizes[i+1], stride=strides[i+1],
                                             padding=paddings[i+1], bias=False))
            layers.append(nn.BatchNorm2d(num_features=hidden_channels[i+1]))
            layers.append(self.g_activation)
        layers.append(nn.ConvTranspose2d(in_channels=hidden_channels[-1], out_channels=self.channels,
                                         kernel_size=kernel_sizes[-1], stride=strides[-1], 
                                         padding=paddings[-1]))
        # layers.append(nn.BatchNorm2d(num_features=self.channels))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def create_discriminator(self):
        layers = []
        hidden_channels = self.config['d_hidden_channels']
        kernel_sizes = self.config['d_kernel_sizes']
        strides = self.config['d_strides']
        paddings = self.config['d_paddings']
        in_channels = self.channels
        # image_size = self.image_size

        for i in range(len(hidden_channels)):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels[i],
                                    kernel_size=kernel_sizes[i], stride=strides[i],
                                    padding=paddings[i], bias=False))
            layers.append(nn.BatchNorm2d(num_features=hidden_channels[i]))
            layers.append(self.d_activation)
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
    
    # def weights_init(self, m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         nn.init.normal_(m.weight.data, 0.0, 0.02)
    #     elif classname.find('BatchNorm') != -1:
    #         nn.init.normal_(m.weight.data, 1.0, 0.02)
    #         nn.init.constant_(m.bias.data, 0)

    # def learn(self, dataloader: DataLoader, log_dir=None):
    #     # Create batch of latent vectors that we will use to visualize
    #     # the progression of the generator
    #     fixed_noise = self.sample_z(batch_size=64)

    #     # Establish convention for real and fake labels during training
    #     real_label = 1.
    #     fake_label = 0.

    #     # Lists to keep track of progress
    #     img_list = []
    #     g_losses = []
    #     d_losses = []

    #     print("Starting Training Loop...")
    #     for epoch in range(self.epochs):
    #         for step, (real_image, _) in enumerate(dataloader):
    #             real_image = real_image.to(self.device)
    #             batch_size = real_image.shape[0]
    #             # Update Discriminator
    #             d_loss, gen_image, real_score, fake_score_before_update = self.discriminator_step(
    #                 real_image=real_image, 
    #                 real_label=real_label, 
    #                 fake_label=fake_label,
    #                 batch_size=batch_size
    #             )

    #             # Update Generator
    #             g_loss, fake_score_after_update = self.generator_step(
    #                 gen_image=gen_image, 
    #                 real_label=real_label,
    #                 batch_size=batch_size
    #             )
                
    #             # Output training stats
    #             if step % 50 == 0:
    #                 print(
    #                     f'[{epoch}/{self.epochs}][{step}/{len(dataloader)}]\t'
    #                     f'Loss_D: {d_loss.item():.4f}\tLoss_G: {g_loss.item():.4f}\t'
    #                     f'D(x): {real_score:.4f}\tD(G(z)): {fake_score_before_update:.4f} / {fake_score_after_update:.4f}'
    #                 )
    #             # Save Losses for plotting later
    #             g_losses.append(g_loss.item())
    #             d_losses.append(d_loss.item())
            
    #         with torch.no_grad():
    #             gen_image = self.generator(fixed_noise).detach().cpu()
    #             img = make_grid(gen_image, padding=2, normalize=True)
                
    #             samples_dir = os.path.join(log_dir, 'samples')
    #             os.makedirs(samples_dir, exist_ok=True)
                
    #             sample_path = os.path.join(samples_dir, f'epoch_{epoch}_sample.png')
    #             save_image(img, sample_path)

    #             img_list.append(img)
            
    #         if epoch % 10 == 0:
    #             models_dir = os.path.join(log_dir, 'models')
    #             os.makedirs(models_dir, exist_ok=True)

    #             model_path = os.path.join(models_dir, f'epoch_{epoch}_model.pth')
    #             torch.save(self.state_dict(), model_path)

    #     models_dir = os.path.join(log_dir, 'models')
    #     os.makedirs(models_dir, exist_ok=True)
    #     torch.save(self.state_dict(), os.path.join(models_dir, 'final_model.pth'))
    #     self.plot_loss_curves(g_losses=g_losses, d_losses=d_losses, log_dir=log_dir)
    #     self.visualize_progression(img_list=img_list, dataloader=dataloader, log_dir=log_dir)
        

    # def discriminator_step(self, real_image, real_label, fake_label, batch_size):
    #     """
    #         Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
    #     """
    #     ## Train with all-real batch
    #     self.discriminator.zero_grad()
    #     real_label = torch.full((batch_size, ), real_label, dtype=torch.float, device=self.device)
    #     output = self.discriminator(real_image).view(-1)
    #     real_loss = self.criterion(output, real_label)
    #     real_loss.backward()
    #     real_score = output.mean().item()

    #     ## Train with all-fake batch
    #     z = self.sample_z(batch_size=batch_size)
    #     gen_image = self.generator(z)
    #     fake_label = torch.full((batch_size, ), fake_label, dtype=torch.float, device=self.device)
    #     output = self.discriminator(gen_image.detach()).view(-1)
    #     fake_loss = self.criterion(output, fake_label)
    #     fake_loss.backward()
    #     fake_score_before_update = output.mean().item()
    #     d_loss = real_loss + fake_loss
    #     self.d_optimizer.step()

    #     return d_loss, gen_image, real_score, fake_score_before_update
    
    # def generator_step(self, gen_image, real_label, batch_size):
    #     """
    #         Update Generator: maximize log(D(G(z)))
    #     """
    #     self.generator.zero_grad()
    #     real_label = torch.full((batch_size, ), real_label, dtype=torch.float, device=self.device)  # fake labels are real for generator cost
    #     output = self.discriminator(gen_image).view(-1)
    #     g_loss = self.criterion(output, real_label)
    #     g_loss.backward()
    #     fake_score_after_update = output.mean().item()
    #     self.g_optimizer.step()
        
    #     return g_loss, fake_score_after_update   
    
    def sample_z(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)

        return z
    
    def sample(self, z=None, batch_size=None):
        z = self.sample_z(batch_size=batch_size) if z == None else z
        while len(z.shape) < 4:
            z = z.unsqueeze(-1)
        output = self.generator(z)

        return output
    
    # def plot_loss_curves(self, g_losses, d_losses, log_dir):
    #     """
    #         Loss versus training iteration
    #     """
    #     plt.figure(figsize=(10, 5))
    #     plt.title("Generator and Discriminator Loss During Training")
    #     plt.plot(g_losses,label="Generator")
    #     plt.plot(d_losses,label="Discriminator")
    #     plt.xlabel("iterations")
    #     plt.ylabel("Loss")
    #     plt.legend()

    #     loss_fig_path = os.path.join(log_dir, 'loss.png')
    #     plt.savefig(loss_fig_path)

    #     plt.show()
    #     plt.close()

    # def visualize_progression(self, img_list, dataloader, log_dir):
    #     """
    #         Visualize the training progression of Generator with an animation.
    #     """
    #     fig = plt.figure(figsize=(8, 8))
    #     plt.axis("off")
    #     ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    #     ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    #     ani_path = os.path.join(log_dir, 'progression.gif')
    #     ani.save(ani_path, writer='imagemagick')
        
    #     HTML(ani.to_jshtml())

    #     # Grab a batch of real images from the dataloader
    #     real_batch = next(iter(dataloader))

    #     # Plot the real images
    #     plt.figure(figsize=(15, 8))
    #     plt.subplot(1, 2, 1)
    #     plt.axis("off")
    #     plt.title("Real Images")
    #     plt.imshow(np.transpose(make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1, 2, 0)))

    #     # Plot the Generated images from the last epoch
    #     plt.subplot(1, 2, 2)
    #     plt.axis("off")
    #     plt.title("Gen Images")
    #     plt.imshow(np.transpose(img_list[-1],(1, 2, 0)))

    #     comparision_fig_path = os.path.join(log_dir, 'comparision.png')
    #     plt.savefig(comparision_fig_path)

    #     plt.show()

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