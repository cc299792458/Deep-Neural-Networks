"""
    Wasserstein Generative Adversarial Networks - (Clipping)
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

from gan.deep_convolutional_gan import DCGAN, Discriminator

class Discriminator(Discriminator):
    def __init__(self, config, feature_size):
        super().__init__(config, feature_size)
        # Remove the final Sigmoid layer.
        self.model = nn.Sequential(*list(self.model.children())[:-1])

class WGAN(DCGAN):
    def __init__(self, 
                feature_size: int, 
                config: dict = None, 
                device: str = 'cpu', 
                lr: float = 0.0002, 
                betas: tuple[float, float] = (0.5, 0.999), 
                epochs: int = 50) -> None:
        
        default_config = {
            'weight_clipping_limit': 0.01,
            'n_critic': 5,
            'g_hidden_channels': [1024, 512, 256, 128],
            'g_kernel_sizes': [4, 4, 4, 4, 1],
            'g_strides': [1, 2, 2, 2, 1],
            'g_paddings': [0, 1, 1, 1, 2],
            'discriminator_cls': Discriminator,
            'd_hidden_channels': [256, 512, 1024],
            'd_kernel_sizes': [4, 4, 4, 4],
            'd_strides': [2, 2, 2, 1],
            'd_paddings': [1, 1, 2, 0],
        }
        
        if config is not None:
            default_config.update(config)

        super().__init__(feature_size, default_config, device, lr, betas, epochs)

        self.weight_clipping_limit = self.config.get('weight_clipping_limit')
        self.n_critic = self.config.get('n_critic')

        # Loss will be calculated by a formula directly.
        self.criterion = None

        self.g_optimizer = optim.RMSprop(params=self.generator.parameters(), lr=lr)
        self.d_optimizer = optim.RMSprop(params=self.discriminator.parameters(), lr=lr)

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

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.weight_clipping_limit, self.weight_clipping_limit)

                # Train the generator every n_critic iterations
                if step % self.n_critic == 0:
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
        real_loss = -torch.mean(output)
        real_loss.backward()
        real_score = output.mean().item()

        ## Train with all-fake batch
        z = self.sample_z(batch_size=batch_size)
        gen_image = self.generator(z)
        fake_label = torch.full((batch_size, ), fake_label, dtype=torch.float, device=self.device)
        output = self.discriminator(gen_image.detach()).view(-1)
        fake_loss = torch.mean(output)
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
        g_loss = -torch.mean(output)
        g_loss.backward()
        fake_score_after_update = output.mean().item()
        self.g_optimizer.step()
        
        return g_loss, fake_score_after_update   


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
    lr = 2e-4
    epochs = 200

    wgan = WGAN(feature_size=feature_size, device=device,
            config={'latent_dim': latent_dim, 
                    'channels': channels, 
                    'image_size': image_size,}, 
                    lr=lr, epochs=epochs).to(device)

    train = True
    ##### 1. Train the model #####
    if train:
        wgan.learn(dataloader=dataloader, log_dir=log_dir)

    ##### 2. Generate image from random noise #####
    else:
        ## Load Model ##
        model_path = os.path.join(log_dir, 'models/final_model.pth')
        wgan.load_state_dict(torch.load(model_path))

        num_images = 400
        z_ranges = ((-1, 1), (-1, 1))
        generate_random_images_and_save(wgan, 
                                        num_images=num_images, 
                                        log_dir=log_dir, 
                                        image_size=image_size, 
                                        latent_dim=latent_dim)