"""
    Generative Adversarial Networks
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

plt.rcParams['animation.embed_limit'] = 100

class Generator(nn.Module):
    def __init__(self, config, feature_size=None):
        super(Generator, self).__init__()
        self.config = config
        self.feature_size = feature_size
        
        self.model = self.create_generator()
        self.apply(self.weights_init)

    def create_generator(self):
        layers = []
        self.channels = self.config['channels']
        self.image_size = self.config['image_size']
        self.latent_dim = self.config['latent_dim']
        hidden_sizes = self.config['g_hidden_sizes']
        activation = self.config['g_activation']

        input_size = self.latent_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            # layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation)
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.feature_size))
        # TODO: figure out why the network collapse when it uses this BatchNorm1d
        # layers.append(nn.BatchNorm1d(self.feature_size))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(self, z):

        return self.model(z).view(-1, self.channels, self.image_size, self.image_size)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if 'Linear' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        
class Discriminator(nn.Module):
    def __init__(self, config, feature_size):
        super(Discriminator, self).__init__()
        self.config = config
        self.feature_size = feature_size
        self.model = self.create_discriminator()
        self.apply(self.weights_init)

    def forward(self, img):
        img = img.view(-1, self.feature_size)
        
        return self.model(img)

    def create_discriminator(self):
        layers = []
        hidden_sizes = self.config['d_hidden_sizes']
        activation = self.config['d_activation']
                
        input_size = self.feature_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            # TODO: figure out why the network works with Dropout1d but not BatchNorm1d
            # layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout1d(0.3))
            layers.append(activation)
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        # layers.append(nn.BatchNorm1d(1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if 'Linear' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

class GAN(nn.Module):
    def __init__(self,
                feature_size: int,
                config: dict = None,
                device: str = 'cpu', 
                lr: float = 2e-4, 
                betas: tuple[float, float] = (0.5, 0.999),
                epochs: int = 50) -> None:
        super(GAN, self).__init__()
        self.device = device
        self.feature_size = feature_size
        
        self.config = {
            'channels': 1,
            'image_size': 28,   # Default for MNIST
            'latent_dim': 128,
            'generator_cls': Generator,
            'g_hidden_sizes': [256, 512, 1024],
            'g_activation': nn.LeakyReLU(0.2),
            'discriminator_cls': Discriminator, 
            'd_hidden_sizes': [1024, 512, 256],
            'd_activation': nn.LeakyReLU(0.2),
        }

        if config is not None:
            self.config.update(config)

        self.channels = self.config.get('channels')
        self.image_size = self.config.get('image_size')
        self.latent_dim = self.config.get('latent_dim')
        generator_cls = self.config.get('generator_cls')
        discriminator_cls = self.config.get('discriminator_cls')

        # Create generator and discriminator
        self.generator = generator_cls(self.config, self.feature_size).to(device)
        self.discriminator = discriminator_cls(self.config, self.feature_size).to(device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        
        self.criterion = nn.BCELoss()
        self.epochs = epochs    
    
    def learn(self, dataloader: DataLoader, log_dir=None):
        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        fixed_noise = self.sample_z(batch_size=64)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Lists to keep track of progress
        img_list = []
        g_losses = []
        d_losses = []

        print("Starting Training Loop...")
        for epoch in range(self.epochs):
            for step, (real_image, _) in enumerate(dataloader):
                real_image = real_image.to(self.device)
                batch_size = real_image.shape[0]
                # Update Discriminator
                d_loss, gen_image, real_score, fake_score_before_update = self.discriminator_step(
                    real_image=real_image, 
                    real_label=real_label, 
                    fake_label=fake_label,
                    batch_size=batch_size
                )

                # Update Generator
                g_loss, fake_score_after_update = self.generator_step(
                    gen_image=gen_image, 
                    real_label=real_label,
                    batch_size=batch_size
                )
                
                # Output training stats
                if step % 50 == 0:
                    print(
                        f'[{epoch}/{self.epochs}][{step}/{len(dataloader)}]\t'
                        f'Loss_D: {d_loss.item():.4f}\tLoss_G: {g_loss.item():.4f}\t'
                        f'D(x): {real_score:.4f}\tD(G(z)): {fake_score_before_update:.4f} / {fake_score_after_update:.4f}'
                    )
                # Save Losses for plotting later
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            with torch.no_grad():
                gen_image = self.generator(fixed_noise).detach().cpu()
                img = make_grid(gen_image, padding=2, normalize=True)
                
                samples_dir = os.path.join(log_dir, 'samples')
                os.makedirs(samples_dir, exist_ok=True)
                
                sample_path = os.path.join(samples_dir, f'epoch_{epoch}_sample.png')
                save_image(img, sample_path)

                img_list.append(img)
            
            if epoch % 10 == 0:
                models_dir = os.path.join(log_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)

                model_path = os.path.join(models_dir, f'epoch_{epoch}_model.pth')
                torch.save(self.state_dict(), model_path)

        models_dir = os.path.join(log_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_dir, 'final_model.pth'))
        self.plot_loss_curves(g_losses=g_losses, d_losses=d_losses, log_dir=log_dir)
        self.visualize_progression(img_list=img_list, dataloader=dataloader, log_dir=log_dir)
        

    def discriminator_step(self, real_image, real_label, fake_label, batch_size):
        """
            Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        """
        ## Train with all-real batch
        self.discriminator.zero_grad()
        real_label = torch.full((batch_size, ), real_label, dtype=torch.float, device=self.device)
        output = self.discriminator(real_image).view(-1)
        real_loss = self.criterion(output, real_label)
        real_loss.backward()
        real_score = output.mean().item()

        ## Train with all-fake batch
        z = self.sample_z(batch_size=batch_size)
        gen_image = self.generator(z)
        fake_label = torch.full((batch_size, ), fake_label, dtype=torch.float, device=self.device)
        output = self.discriminator(gen_image.detach()).view(-1)
        fake_loss = self.criterion(output, fake_label)
        fake_loss.backward()
        fake_score_before_update = output.mean().item()
        d_loss = real_loss + fake_loss
        self.d_optimizer.step()

        return d_loss, gen_image, real_score, fake_score_before_update
    
    def generator_step(self, gen_image, real_label, batch_size):
        """
            Update Generator: maximize log(D(G(z)))
        """
        self.generator.zero_grad()
        real_label = torch.full((batch_size, ), real_label, dtype=torch.float, device=self.device)  # fake labels are real for generator cost
        output = self.discriminator(gen_image).view(-1)
        g_loss = self.criterion(output, real_label)
        g_loss.backward()
        fake_score_after_update = output.mean().item()
        self.g_optimizer.step()
        
        return g_loss, fake_score_after_update   
    
    def sample_z(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, device=self.device)

        return z
    
    def sample(self, z=None, batch_size=None):
        z = self.sample_z(batch_size=batch_size) if z == None else z
        output = self.generator(z)

        return output
    
    def plot_loss_curves(self, g_losses, d_losses, log_dir):
        """
            Loss versus training iteration
        """
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses,label="Generator")
        plt.plot(d_losses,label="Discriminator")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()

        loss_fig_path = os.path.join(log_dir, 'loss.png')
        plt.savefig(loss_fig_path)

        plt.show()
        plt.close()

    def visualize_progression(self, img_list, dataloader, log_dir):
        """
            Visualize the training progression of Generator with an animation.
        """
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        ani_path = os.path.join(log_dir, 'progression.gif')
        ani.save(ani_path, writer='imagemagick')
        
        HTML(ani.to_jshtml())

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(dataloader))

        # Plot the real images
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1, 2, 0)))

        # Plot the Generated images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Gen Images")
        plt.imshow(np.transpose(img_list[-1],(1, 2, 0)))

        comparision_fig_path = os.path.join(log_dir, 'comparision.png')
        plt.savefig(comparision_fig_path)

        plt.show()

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
    epochs = 200

    gan = GAN(feature_size=feature_size, device=device,
            config={'latent_dim': latent_dim, 
                    'channels': channels, 
                    'image_size': image_size,}, epochs=epochs).to(device)

    train = True
    ##### 1. Train the model #####
    if train:
        gan.learn(dataloader=dataloader, log_dir=log_dir)

    ##### 2. Generate image from random noise #####
    else:
        ## Load Model ##
        gan.load_state_dict(torch.load(os.path.join(log_dir, f'/models/final_model.pth')))

        num_images = 400
        z_ranges = ((-1, 1), (-1, 1))
        generate_random_images_and_save(gan, 
                                        num_images=num_images, 
                                        log_dir=log_dir, 
                                        image_size=image_size, 
                                        latent_dim=latent_dim)