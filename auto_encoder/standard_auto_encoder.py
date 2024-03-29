"""
    Implement a most trivial auto-encoder to for MNIST dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from itertools import chain
from tqdm import tqdm

import os

class AutoEncoder(nn.Module):
    def __init__(self, feature_size, device='cpu', lr=1e-3, epochs=20) -> None:
        super(AutoEncoder, self).__init__()
        self.device = device
        self.feature_size = feature_size
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr)
        self.loss_function = nn.MSELoss()
        self.epochs = epochs

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder(embedding)

        return output
    
    def sample_and_save_image(self, dataloader, log_dir):
        fixed_sample, _ = next(iter(dataloader))
        fixed_sample_image = to_pil_image(fixed_sample[0])
        save_path = os.path.join(log_dir, 'fixed_sample.png')
        fixed_sample_image.save(save_path)
        fixed_sample = fixed_sample[0].reshape(1, -1).to(self.device)
        return fixed_sample

    def reconstruct_and_save_image(self, sample, log_dir, channels, image_size, epoch=None):
        epoch = 'init' if epoch is None else f'epoch_{epoch}'
        with torch.no_grad():
            reconstructed_sample = self.forward(sample).reshape(channels, image_size, image_size)
            reconstructed_image = to_pil_image(reconstructed_sample.cpu())
            save_path = os.path.join(log_dir, f'reconstructed_image_{epoch}.png')
            reconstructed_image.save(save_path)
    
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
                batch = batch.to(self.device).reshape(-1, self.feature_size)
                self.optimizer.zero_grad()

                output = self.forward(batch)
                loss = self.loss_function(output, batch)

                loss.backward()
                self.optimizer.step()
                
                if step % 50 == 0:
                    loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss=loss.item())
    
            if loss.item() < best_loss - delta:
                best_loss = loss.item()
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
    ##### 0. Load MNIST dataset #####
    dataset_name = 'CIFAR-10'
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
    auto_encoder = AutoEncoder(feature_size=feature_size ,device=device).to(device)
    auto_encoder.learn(dataloader=dataloader, log_dir=log_dir, channels=channels, image_size=image_size)
