"""
    Implement a most trivial auto-encoder to for MNIST dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from tqdm import tqdm

import os

class AutoEncoder(nn.Module):
    def __init__(self, 
                 feature_size: int,
                 config: dict = None,
                 device: str = 'cpu', 
                 lr: float = 1e-3, 
                 epochs: int = 20) -> None:
        super(AutoEncoder, self).__init__()
        self.device = device
        self.feature_size = feature_size
        self.config = {
            'type': 'fc',
            'latent_dim': 16,
            'hidden_sizes': [128, 64, 32],
            'channels': 1,
            'image_size': 28,   # Default for MNIST
            'encoder_hidden_channels': [16, 32],    # Used for CNN #
            'encoder_kernel_sizes': [3, 3],
            'encoder_strides': [2, 2],
            'encoder_paddings': [1, 1],
            'decoder_hidden_channels': [32, 16],    # Used for CNN #
            'decoder_kernel_sizes': [3, 3],
            'decoder_strides': [2, 2],
            'decoder_paddings': [1, 1],
            'decoder_output_paddings': [1, 1]
        }

        if config is not None:
            self.config.update(config)

        self.architecture_type = self.config.get('type', 'fc')

        if self.architecture_type == 'fc':
            self.encoder = self.create_fc_encoder()
            self.decoder = self.create_fc_decoder()
        elif self.architecture_type == 'cnn':
            self.encoder = self.create_cnn_encoder()
            self.decoder = self.create_cnn_decoder()
        else:
            raise ValueError("Unsupported architecture type")

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.epochs = epochs

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder(embedding)

        return output
    
    def calc_loss(self, input, output):
        return F.mse_loss(input, output)
    
    def create_fc_encoder(self):
        layers = []
        input_size = self.feature_size
        for hidden_size in self.config['hidden_sizes']:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.config['latent_dim']))
        return nn.Sequential(*layers)

    def create_fc_decoder(self):
        layers = []
        hidden_sizes = list(reversed(self.config['hidden_sizes']))
        input_size = self.config['latent_dim']
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.feature_size))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    def create_cnn_encoder(self):
        input_channels = self.config['channels']
        image_size = self.config['image_size']
        encoder_hidden_channels = self.config['encoder_hidden_channels']
        encoder_kernel_sizes = self.config['encoder_kernel_sizes']
        encoder_strides = self.config['encoder_strides']
        encoder_paddings = self.config['encoder_paddings']
        latent_dim = self.config['latent_dim']

        layers = []
        in_channels = input_channels
        for out_channels, kernel_size, stride, padding in zip(encoder_hidden_channels, encoder_kernel_sizes, encoder_strides, encoder_paddings):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU()
            ]
            in_channels = out_channels
            image_size = (image_size - kernel_size + 2 * padding) // stride + 1

        layers += [
            nn.Flatten(),
            nn.Linear(in_channels * image_size * image_size, latent_dim)
        ]

        self.reduced_size = image_size

        return nn.Sequential(*layers)

    def create_cnn_decoder(self):
        latent_dim = self.config['latent_dim']
        decoder_hidden_channels = self.config['decoder_hidden_channels']
        decoder_kernel_sizes = self.config['decoder_kernel_sizes']
        decoder_strides = self.config['decoder_strides']
        decoder_paddings = self.config['decoder_paddings']
        decoder_output_paddings = self.config['decoder_output_paddings']
        output_channels = self.config['channels']

        layers = [nn.Linear(latent_dim, decoder_hidden_channels[0] * self.reduced_size * self.reduced_size), nn.ReLU()]
        layers += [nn.Unflatten(1, (decoder_hidden_channels[0], self.reduced_size, self.reduced_size))]

        for i in range(len(decoder_hidden_channels) - 1):
            layers += [
                nn.ConvTranspose2d(decoder_hidden_channels[i], decoder_hidden_channels[i+1], 
                                kernel_size=decoder_kernel_sizes[i], 
                                stride=decoder_strides[i], 
                                padding=decoder_paddings[i], 
                                output_padding=decoder_output_paddings[i]),
                nn.ReLU()
            ]

        layers += [
            nn.ConvTranspose2d(decoder_hidden_channels[-1], output_channels, 
                            kernel_size=decoder_kernel_sizes[-1], 
                            stride=decoder_strides[-1], 
                            padding=decoder_paddings[-1], 
                            output_padding=decoder_output_paddings[-1]),
            nn.Sigmoid()
        ]

        return nn.Sequential(*layers)

    def sample_and_save_image(self, dataloader, log_dir):
        fixed_sample, _ = next(iter(dataloader))
        fixed_sample = fixed_sample[0:1]
        fixed_sample_image = to_pil_image(fixed_sample[0])
        save_path = os.path.join(log_dir, 'fixed_sample.png')
        fixed_sample_image.save(save_path)
        if self.architecture_type == 'fc':
            fixed_sample = fixed_sample[0].reshape(1, -1).to(self.device)
        else:
            fixed_sample = fixed_sample.to(self.device)
        return fixed_sample
    

    def reconstruct_and_save_image(self, sample, log_dir, channels, image_size, epoch=None):
        epoch = 'init' if epoch is None else f'epoch_{epoch}'
        with torch.no_grad():
            reconstructed_sample = self.forward(sample).reshape(channels, image_size, image_size)
            reconstructed_image = to_pil_image(reconstructed_sample.cpu())
            save_path = os.path.join(log_dir, f'reconstructed_image_{epoch}.png')
            reconstructed_image.save(save_path)
    
    def step(self, batch):
        self.optimizer.zero_grad()
        outputs = self.forward(batch)
        loss = self.calc_loss(batch, outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def learn(self, dataloader, log_dir, channels, image_size, patience=8, delta=0):
        fixed_sample = self.sample_and_save_image(dataloader=dataloader, log_dir=log_dir)
        self.eval()
        self.reconstruct_and_save_image(fixed_sample, log_dir, channels, image_size)
        
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(self.epochs):
            model_saved_this_epoch = False

            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
            self.train()
            for step, (batch, _) in loop:
                if self.architecture_type == 'fc':
                    batch = batch.to(self.device).reshape(-1, self.feature_size)
                else:
                    batch = batch.to(self.device)

                loss_item = self.step(batch=batch)
                
                if step % 50 == 0:
                    loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss=loss_item)
    
            if loss_item < best_loss - delta:
                best_loss = loss_item
                torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.pth'))
                model_saved_this_epoch = True
                patience_counter = 0
            else:
                patience_counter += 1

            self.eval()
            self.reconstruct_and_save_image(fixed_sample, log_dir, channels, image_size, epoch=epoch)

            if patience_counter >= patience:
                print(f"No improvement in validation loss for {patience} consecutive epochs. Stopping early.")
                break

            if model_saved_this_epoch:
                print(f"New best model saved with loss {best_loss}")

if __name__ == '__main__':
    ##### 0. Load Dataset #####
    dataset_name = 'MNIST'
    if dataset_name == 'MNIST':
        dataset = MNIST(root='./data', transform=ToTensor(), download=True)
        channels = 1
        image_size = 28
    elif dataset_name == 'CIFAR-10':
        dataset = CIFAR10(root='./data', transform=ToTensor(), download=True)
        channels = 3
        image_size = 32
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    ##### 1. Train the autoencoder #####
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/{dataset_name}/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    feature_size = channels * image_size * image_size
    auto_encoder = AutoEncoder(feature_size=feature_size, device=device,
                               config={'type': 'cnn', 
                                       'hidden_sizes': [512, 256, 128, 64, 32],
                                       'channels': channels,
                                       'image_size': image_size,}).to(device)
    auto_encoder.learn(dataloader=dataloader, log_dir=log_dir, channels=channels, image_size=image_size)

    ##### 2. Generate image from random noise #####