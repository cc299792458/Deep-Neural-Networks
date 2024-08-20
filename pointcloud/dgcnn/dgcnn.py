import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.data import Data, DataLoader

# TODO: Parameterize the network with the network architecture

class DGCNN(nn.Module):
    def __init__(self, k=20, num_classes=10):
        super(DGCNN, self).__init__()
        self.k = k
        self.conv1 = DynamicEdgeConv(nn.Linear(6, 64), k=self.k)
        self.conv2 = DynamicEdgeConv(nn.Linear(128, 128), k=self.k)
        self.conv3 = DynamicEdgeConv(nn.Linear(256, 256), k=self.k)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        pass