"""
    Implement a most trivial auto-encoder to for MNIST dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from itertools import chain

class AutoEncoder(nn.Module):
    def __init__(self, device='cpu', lr=1e-3, epochs=20) -> None:
        super(AutoEncoder, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
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
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr)
        self.loss_function = nn.MSELoss()
        self.epochs = epochs

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder(embedding)

        return output
    
    def train(self, dataloader):
        for epoch in range(self.epochs):
            for step, (batch, _) in enumerate(dataloader):
                batch = batch.to(self.device).reshape(-1, 28 * 28)
                self.optimizer.zero_grad()

                output = self.forward(batch)
                loss = self.loss_function(output, batch)

                loss.backward()
                self.optimizer.step()

                if epoch % 5 == 0 and step == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()}")



if __name__ == '__main__':
    ##### 0. Load MNIST dataset #####
    dataset = torchvision.datasets.MNIST(root='./data', transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 32, shuffle = True)

    ##### 1. Train the autoencoder #####
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    auto_encoder = AutoEncoder(device=device).to(device)
    auto_encoder.train(dataloader=dataloader)

    ##### 2. Evaluate the autoencoder #####
