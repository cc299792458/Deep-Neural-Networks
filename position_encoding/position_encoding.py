import math

import torch
import torch.nn as nn

#NOTE: Not sure this form is absolutly the same to the pos encoding in "Attention is All You Need"
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        
        self.dim = dim

    def forward(self, time):
        # Calculate the scale
        device = time.device
        half_dim = self.dim // 2
        scale = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * (-math.log(10000.0) / half_dim))
        
        # Calculate the time encodings
        time = time.unsqueeze(1)
        time_encodings = torch.zeros(time.size(0), self.dim, device=device)
        time_encodings[:, 0::2] = torch.sin(time * scale)
        time_encodings[:, 1::2] = torch.cos(time * scale)
        
        return time_encodings
    
if __name__ == '__main__':
    position_embeddings = SinusoidalPositionEmbeddings()