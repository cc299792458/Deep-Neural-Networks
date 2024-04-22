"""
    Conditional Generative Adversarial Networks (CGAN)
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

from utils.misc_utils import set_seed, plot_data_from_dataloader, generate_conditional_images_and_save

from gan.deep_convolutional_gan import DCGAN, Generator, Discriminator

class Generator(Generator):
    def __init__(self, config, feature_size=None):
        super().__init__(config, feature_size)

        self.latent_dim = self.config['latent_dim']
        self.num_classes = self.config['num_classes']
        self.label_emb_dim = self.config['label_emb_dim'] if self.config.get('label_emb_dim') is not None else self.num_classes
        
        self.modify_first_layer()
        self.label_emb = nn.Embedding(self.num_classes, self.label_emb_dim)
    
    def forward(self, z, label):
        # Embed label
        label = self.label_emb(label)
        label = label.view(label.size(0), label.size(1), 1, 1)
        # Concatenate z and label
        input = torch.cat([z, label], 1)
        
        return self.model(input)

    def modify_first_layer(self):
        """
            Modify the first layer to accommodate label_embedding.
        """
        first_layer = list(self.model.children())[0]
        modified_first_layer = nn.ConvTranspose2d(self.latent_dim + self.label_emb_dim, first_layer.out_channels,
                                                  kernel_size=first_layer.kernel_size, 
                                                  stride=first_layer.stride,
                                                  padding=first_layer.padding)
        new_layers = [modified_first_layer] + list(self.model.children())[1:]

        self.model = nn.Sequential(*new_layers)
    

class Discriminator(Discriminator):
    def __init__(self, config, feature_size):
        super().__init__(config, feature_size)

        self.image_size = self.config['image_size']
        self.latent_dim = self.config['latent_dim']
        self.num_classes = self.config['num_classes']
        self.label_emb_dim = self.config['label_emb_dim'] if self.config.get('label_emb_dim') is not None else self.num_classes
        
        self.modify_first_layer()
        self.label_emb = nn.Embedding(self.num_classes, self.label_emb_dim)

    def forward(self, img, label):
        # Expand label to match image size and concatenate
        label = self.label_emb(label)
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.expand(-1, -1, self.image_size, self.image_size)
        img = torch.cat([img, label], 1)

        return self.model(img)

    def modify_first_layer(self):
        """
            Modify the first layer to accommodate label_embedding.
        """
        first_layer = list(self.model.children())[0]
        modified_first_layer = nn.Conv2d(first_layer.in_channels + self.label_emb_dim, first_layer.out_channels,
                                         kernel_size=first_layer.kernel_size, 
                                         stride=first_layer.stride,
                                         padding=first_layer.padding)
        new_layers = [modified_first_layer] + list(self.model.children())[1:]

        self.model = nn.Sequential(*new_layers)

class CGAN(DCGAN):
    """
        Conditional Generative Adversarial Networks
    """

    def __init__(self, 
                feature_size: int, 
                config: dict = None, 
                device: str = 'cpu', lr: float = 0.0002, 
                betas: tuple[float, float] = (0.5, 0.999), 
                epochs: int = 50) -> None:
        default_config = {
            'num_classes': 10,
            # 'label_emb_dim': 10,
            'generator_cls': Generator,
            'discriminator_cls': Discriminator,
        }
        
        if config is not None:
            default_config.update(config)
        
        super().__init__(feature_size, default_config, device, lr, betas, epochs)

        self.num_classes = self.config.get('num_classes')
        # self.label_emb_dim = self.config.get('label_emb_dim') if self.config.get('label_emb_dim') is not None else self.num_classes
        # self.label_emb = nn.Embedding(self.num_classes, self.label_emb_dim)

        # self.modify_generator()
        # self.modify_discriminator()
        # self.generator.apply(self.weights_init)
        # self.discriminator.apply(self.weights_init)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

    def learn(self, dataloader: DataLoader, log_dir=None):
        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        fixed_z, fixed_label = self.sample_z_and_label(num_per_cls=8, deterministic=True)

        # Establish convention for real and fake labels during training
        real_label = 1.
        wrong_label = 0.
        fake_label = 0.

        # Lists to keep track of progress
        img_list = []
        g_losses = []
        d_losses = []

        print("Starting Training Loop...")
        for epoch in range(self.epochs):
            for step, (real_image, correct_label) in enumerate(dataloader):
                real_image = real_image.to(self.device)
                correct_label = correct_label.to(self.device)
                batch_size = real_image.shape[0]
                # Update Discriminator
                d_loss, gen_image, gen_label, real_score, incorrect_score, fake_score_before_update = self.discriminator_step(
                    real_image=real_image, 
                    correct_label=correct_label,
                    real_label=real_label, 
                    wrong_label=wrong_label,
                    fake_label=fake_label,
                    batch_size=batch_size
                )

                # Update Generator
                g_loss, fake_score_after_update = self.generator_step(
                    gen_image=gen_image, 
                    gen_label=gen_label,
                    real_label=real_label,
                    batch_size=batch_size
                )
                
                # Output training stats
                if step % 50 == 0:
                    print(
                        f'[{epoch}/{self.epochs}][{step}/{len(dataloader)}] '
                        f'Loss_D: {d_loss:.4f}, Loss_G: {g_loss:.4f}, '
                        f'D(x): {real_score:.4f}, D(x_wrong): {incorrect_score:.4f}, '
                        f'D(G(z)): {fake_score_before_update:.4f} / {fake_score_after_update:.4f}'
                    )
                # Save Losses for plotting later
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            with torch.no_grad():
                gen_image = self.generator(fixed_z, fixed_label).detach().cpu()
                img = make_grid(gen_image, nrow=10, padding=2, normalize=True)
                
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
        
    def discriminator_step(self, real_image, correct_label, real_label, wrong_label, fake_label, batch_size):
        """
            Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        """
        ## Train with real image, correct label
        self.discriminator.zero_grad()
        real_label = torch.full((batch_size, ), real_label, dtype=torch.float, device=self.device)
        output = self.discriminator(real_image, correct_label).view(-1)
        real_loss = self.criterion(output, real_label)
        real_loss.backward()
        real_score = output.mean().item()

        ## Train with real image, incorrect label
        wrong_label = torch.full((batch_size, ), wrong_label, dtype=torch.float, device=self.device)
        incorrect_label = self.get_incorrect_label(correct_label=correct_label)
        output = self.discriminator(real_image, incorrect_label).view(-1)
        incorrect_loss = self.criterion(output, wrong_label)
        incorrect_loss.backward()
        incorrect_score = output.mean().item()

        ## Train with fake image
        z, gen_label = self.sample_z_and_label(batch_size=batch_size, deterministic=False)
        gen_image = self.generator(z, gen_label)
        fake_label = torch.full((batch_size, ), fake_label, dtype=torch.float, device=self.device)
        output = self.discriminator(gen_image.detach(), gen_label.detach()).view(-1)
        fake_loss = self.criterion(output, fake_label)
        fake_loss.backward()
        fake_score_before_update = output.mean().item()
        d_loss = real_loss + incorrect_loss + fake_loss
        self.d_optimizer.step()

        return d_loss, gen_image, gen_label, real_score, incorrect_score, fake_score_before_update
    
    def generator_step(self, gen_image, gen_label, real_label, batch_size):
        """
            Update Generator: maximize log(D(G(z)))
        """
        self.generator.zero_grad()
        real_label = torch.full((batch_size, ), real_label, dtype=torch.float, device=self.device)  # fake labels are real for generator cost
        output = self.discriminator(gen_image, gen_label).view(-1)
        g_loss = self.criterion(output, real_label)
        g_loss.backward()
        fake_score_after_update = output.mean().item()
        self.g_optimizer.step()
        
        return g_loss, fake_score_after_update
    
    def get_incorrect_label(self, correct_label):
        random_addition = torch.randint(low=1, high=self.num_classes, size=(correct_label.size(0),), device=correct_label.device)
        incorrect_label = (correct_label + random_addition) % self.num_classes

        return incorrect_label

    def sample_z_and_label(self, batch_size=None, num_per_cls=None, deterministic=True):
        if not deterministic:
            z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            label = torch.randint(low=0, high=self.num_classes, size=(batch_size,), device=self.device)
        else:
            batch_size = num_per_cls * self.num_classes
            z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            label = torch.arange(self.num_classes, device=self.device).repeat(num_per_cls)

        return z, label
    
    def sample(self, z=None, label=None, batch_size=None, num_per_cls=None, deterministic=True):
        if z is None or label is None:
            z, label = self.sample_z_and_label(batch_size=batch_size, num_per_cls=num_per_cls, deterministic=deterministic)
        output = self.generator(z, label)

        return output
    
    
if __name__ == '__main__':
    set_seed()
    ##### 0. Load Dataset #####
    dataset_name = 'MNIST'
    batch_size = 128

    if dataset_name == 'MNIST':
        channels = 1
        image_size = 28
        num_classes = 10
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
    epochs = 50

    cgan = CGAN(feature_size=feature_size, device=device,
            config={'latent_dim': latent_dim, 
                    'channels': channels, 
                    'image_size': image_size,
                    'num_classes': num_classes,}, 
                    lr=lr, epochs=epochs).to(device)

    train = False
    ##### 1. Train the model #####
    if train:
        cgan.learn(dataloader=dataloader, log_dir=log_dir)

    ##### 2. Generate image from random noise #####
    else:
        ## Load Model ##
        model_path = os.path.join(log_dir, 'models/final_model.pth')
        cgan.load_state_dict(torch.load(model_path))

        num_per_cls = 20
        generate_conditional_images_and_save(model=cgan,
                                             num_per_cls=num_per_cls,
                                             num_classes=num_classes,
                                             log_dir=log_dir,
                                             image_size=image_size,
                                             latent_dim=latent_dim)