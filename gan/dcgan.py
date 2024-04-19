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
from torchvision.utils import make_grid
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

from IPython.display import HTML
from utils.misc_utils import set_seed, plot_data_from_dataloader

class DCGAN(nn.Module):
    def __init__(self,
                feature_size: int,
                config: dict = None,
                device: str = 'cpu', 
                lr: float = 2e-4, 
                betas: tuple[float, float] = (0.5, 0.999),
                epochs: int = 50) -> None:
        super(DCGAN, self).__init__()
        self.device = device
        self.feature_size = feature_size
        self.config = {
            'channels': 1,
            'image_size': 28,   # Default for MNIST
            'latent_dim': 2,
            ## For Fully Connected Network ##
            'generator_hidden_sizes': [128, 256, 512, 1024],
            'discriminator_hidden_sizes': [512, 256],
            ## For Convolutional Network ##
            'g_hidden_channels': [128, 128, 64],
            'g_kernel_sizes': [4, 4, 3],
            'g_strides': [2, 2, 1],
            'g_paddings': [1, 1, 1],
            'd_hidden_channels': [16, 32, 64, 128],
            'd_kernel_sizes': [3, 3, 3, 3],
            'd_strides': [2, 2, 2, 2],
            'd_paddings': [1, 1, 1, 1],
            'activation': nn.LeakyReLU(0.2)
        }

        if config is not None:
            self.config.update(config)

        self.channels = self.config.get('channels')
        self.image_size = self.config.get('image_size')
        self.latent_dim = self.config.get('latent_dim')
        self.architecture_type = self.config.get('type')
        self.activation = self.config.get('activation')

        self.generator = self.create_cnn_generator()
        self.discriminator = self.create_cnn_discriminator()
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.criterion = nn.BCELoss()
        self.epochs = epochs

    def create_cnn_generator(self):
        layers = []
        init_size = self.image_size // 4   # 28 // 4 = 7
        hidden_channels = self.config['g_hidden_channels']
        kernel_sizes = self.config['g_kernel_sizes']
        strides = self.config['g_strides']
        paddings = self.config['g_paddings']

        layers.append(nn.Linear(self.latent_dim, hidden_channels[0] * init_size * init_size))
        # Seems like Unflatten can be replaced by a ConvTranspose2d
        layers.append(nn.Unflatten(1, (hidden_channels[0], init_size, init_size)))
        layers.append(nn.BatchNorm2d(hidden_channels[0]))
        for i in range(len(hidden_channels) - 1):
            layers.append(nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i+1],
                                             kernel_size=kernel_sizes[i], stride=strides[i],
                                             padding=paddings[i]))
            layers.append(nn.BatchNorm2d(num_features=hidden_channels[i+1]))
            layers.append(self.activation)
        layers.append(nn.ConvTranspose2d(in_channels=hidden_channels[-1], out_channels=self.channels,
                                         kernel_size=kernel_sizes[-1], stride=strides[-1], 
                                         padding=paddings[-1]))
        layers.append(nn.BatchNorm2d(num_features=self.channels, momentum=0.8))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def create_cnn_discriminator(self):
        layers = []
        hidden_channels = self.config['d_hidden_channels']
        kernel_sizes = self.config['d_kernel_sizes']
        strides = self.config['d_strides']
        paddings = self.config['d_paddings']
        in_channels = self.channels
        image_size = self.image_size

        for i in range(len(hidden_channels)):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels[i],
                                    kernel_size=kernel_sizes[i], stride=strides[i],
                                    padding=paddings[i]))
            layers.append(self.activation)
            layers.append(nn.Dropout2d(0.25))
            layers.append(nn.BatchNorm2d(num_features=hidden_channels[i], momentum=0.8))
            in_channels = hidden_channels[i]
            image_size = (image_size - kernel_sizes[i] + 2 * paddings[i]) // strides[i] + 1
        final_num_features = hidden_channels[-1] * image_size * image_size
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=final_num_features, out_features=1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def learn(self, dataloader, log_dir=None):
        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        fixed_noise = torch.randn(64, latent_dim, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                dcgan.discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = dcgan.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = dcgan.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, latent_dim, device=device)
                # Generate fake image batch with G
                fake = dcgan.generator(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = dcgan.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = dcgan.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                dcgan.d_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                dcgan.generator.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = dcgan.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = dcgan.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                dcgan.g_optimizer.step()
                
                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, self.epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = dcgan.generator(fixed_noise).detach().cpu()
                    img_list.append(make_grid(fake, padding=2, normalize=True))
                    
                iters += 1


        ######################################################################
        # Results
        # -------
        # 
        # Finally, lets check out how we did. Here, we will look at three
        # different results. First, we will see how D and G’s losses changed
        # during training. Second, we will visualize G’s output on the fixed_noise
        # batch for every epoch. And third, we will look at a batch of real data
        # next to a batch of fake data from G.
        # 
        # **Loss versus training iteration**
        # 
        # Below is a plot of D & G’s losses versus training iterations.
        # 

        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


        ######################################################################
        # **Visualization of G’s progression**
        # 
        # Remember how we saved the generator’s output on the fixed_noise batch
        # after every epoch of training. Now, we can visualize the training
        # progression of G with an animation. Press the play button to start the
        # animation.
        # 

        #%%capture
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(dataloader))

        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))
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

    dcgan = DCGAN(feature_size=feature_size, device=device, 
              config={'latent_dim': latent_dim, 
                      'channels': channels, 
                      'image_size': image_size,}).to(device)

    train = True
    # ##### 1. Train the dcgan #####
    if train:
        dcgan.learn(dataloader=dataloader)

    