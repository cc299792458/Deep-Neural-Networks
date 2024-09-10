import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.misc_utils import set_seed

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cpu'):
        super(PositionalEncoding, self).__init__()

        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate positional encodings using sin and cos
        _2i = torch.arange(0, d_model, 2).float()
        div_term = torch.exp((-math.log(10000.0) * _2i / d_model)) # Using exp and log for numerical stability
        pe[:, 0::2] = torch.sin(pos * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(pos * div_term)  # Apply cos to odd indices

        # Register as buffer to avoid it being treated as a parameter
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        # Add positional encodings to input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x

def plot_positional_encoding_heatmap(pe_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(pe_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Encoding Value')
    plt.xlabel('Embedding Dimension (d_model)')
    plt.ylabel('Position')
    plt.title('Positional Encoding Heatmap')
    plt.show()

if __name__ == '__main__':
    set_seed()

    # Create positional encoding and plot heatmap
    d_model = 512
    max_len = 1000
    pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
    pe_matrix = pos_encoding.pe.squeeze(0)  # Remove batch dimension
    plot_positional_encoding_heatmap(pe_matrix)
