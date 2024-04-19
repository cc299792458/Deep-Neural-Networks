import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils.misc_utils import set_seed, generate_random_images_and_save


class GAN(nn.Module):
    def __init__(self,
                feature_size: int,
                config: dict = None,
                device: str = 'cpu', 
                lr: float = 2e-4, 
                betas: tuple[float, float] = (0.5, 0.999),
                epochs: int = 200) -> None:
        super(GAN, self).__init__()
        self.device = device
        self.feature_size = feature_size
        self.config = {
            'type': 'fc',
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

        if self.architecture_type == 'fc':
            self.generator = self.create_fc_generator()
            self.discriminator = self.create_fc_discriminator()
        elif self.architecture_type == 'cnn':
            self.generator = self.create_cnn_generator()
            self.discriminator = self.create_cnn_discriminator()
            self.generator.apply(self.weights_init)
            self.discriminator.apply(self.weights_init)
        else:
            raise ValueError("Unsupported architecture type")

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.epochs = epochs
    
    def create_fc_generator(self):
        layers = []
        hidden_sizes = self.config['generator_hidden_sizes']

        input_size = self.latent_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size, momentum=0.8))
            layers.append(self.activation)
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.feature_size))
        layers.append(nn.BatchNorm1d(self.feature_size, momentum=0.8))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def create_fc_discriminator(self):
        layers = []
        hidden_sizes = self.config['discriminator_hidden_sizes']
        
        input_size = self.feature_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            # layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(self.activation)
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        # layers.append(nn.BatchNorm1d(1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

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
    
    def learn(self, dataloader: DataLoader, log_dir, patience=8, delta=0):
        batch_size = dataloader.batch_size
        real_label = torch.ones((batch_size, 1), device=self.device).detach()
        fake_label = torch.zeros((batch_size, 1), device=self.device).detach()
        
        self.eval()
        generate_random_images_and_save(self, num_images=16, log_dir=log_dir, image_size=self.image_size, latent_dim=self.latent_dim)

        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(self.epochs):
            total_loss = 0
            model_saved_this_epoch = False

            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
            self.train()
            for step, (real_image, _) in loop:
                real_image = real_image.to(self.device)
                if self.architecture_type == 'fc':
                    real_image = real_image.reshape(-1, self.feature_size)

                ## Train Generator ##
                g_loss, gen_image = self.generator_step(real_label=real_label, batch_size=batch_size, retain_graph=True)

                ## Train Discriminator ##
                d_loss = self.discriminator_step(real_image=real_image, gen_image=gen_image, real_label=real_label, fake_label=fake_label)
                
                step_loss = g_loss + d_loss
                total_loss += step_loss
                
                if step % 50 == 0:
                    loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss=step_loss)

            avg_loss = total_loss / len(loop)

            if avg_loss < best_loss - delta:
                best_loss = avg_loss
                torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.pth'))
                model_saved_this_epoch = True
                patience_counter = 0
            else:
                patience_counter += 1

            self.eval()
            generate_random_images_and_save(self, num_images=16, log_dir=log_dir, image_size=self.image_size, latent_dim=self.latent_dim)

            if patience_counter >= patience:
                print(f"No improvement in validation loss for {patience} consecutive epochs. Stopping early.")
                break

            if model_saved_this_epoch:
                print(f"New best model saved with loss {best_loss}")

    def generator_step(self, real_label, batch_size, retain_graph=False):
        self.g_optimizer.zero_grad()
        z = torch.randn((batch_size, self.latent_dim), device=self.device)
        gen_image = self.generator(z)
        g_loss = F.binary_cross_entropy(self.discriminator(gen_image), real_label)
        g_loss.backward(retain_graph=retain_graph)
        self.g_optimizer.step()

        return g_loss.item(), gen_image
    
    def discriminator_step(self, real_image, gen_image, real_label, fake_label, retain_graph=False):
        self.d_optimizer.zero_grad()

        real_loss = F.binary_cross_entropy(self.discriminator(real_image), real_label)
        fake_loss = F.binary_cross_entropy(self.discriminator(gen_image), fake_label)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward(retain_graph=retain_graph)
        self.d_optimizer.step()

        return d_loss.item()

    def sample(self, z):
        output = self.generator(z)
        if self.architecture_type == 'fc':
            output = output.reshape(-1, self.channels, self.image_size, self.image_size)

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
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ## Training parameters ## 
    model_type = 'cnn'
    latent_dim = 128

    gan = GAN(feature_size=feature_size, device=device, 
              config={'type': model_type, 
                      'latent_dim': latent_dim, 
                      'channels': channels, 
                      'image_size': image_size,}).to(device)
    
    train = True
    # ##### 1. Train the gan #####
    if train:
        gan.learn(dataloader=dataloader, log_dir=log_dir)
    
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