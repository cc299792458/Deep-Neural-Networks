"""
    Transformer

    Reference: https://github.com/hyunwoongko/transformer
"""

import torch
import torch.nn as nn

from layer_norm import LayerNorm
from token_embedding import TokenEmbedding
from positional_encoding import PositionalEncoding

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)

        return self.drop_out(tok_emb + pos_emb)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head

        self.attention = ScaledDotProductAttention()

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. dot product with weight matrices
        Q, K, V = self.query_linear(Q), self.key_linear(K), self.value_linear(V)
         
        # 2. split tensor by number of heads
        Q, K, V = self.split(Q), self.split(K), self.split(V)

        # 3. do scaled dot product to compute similarity
        out, score = self.attention(Q, K, V, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.out_linear(out)

        return out

    def split(self, x):
        """
        split tensor by number of head
        """
        batch_size, seq_len, d_model = x.size()
        d_head = d_model // self.n_head

        x = x.view(batch_size, seq_len, self.n_head, d_head).transpose(1, 2)

        return x
    
    def concat(self, x):
        batch_size, n_head, seq_len, d_head = x.size()
        d_model = n_head * d_head

        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return x
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        batch_size, n_head, seq_len, d_head = K.size()

        K_T = K.transpose(-2, -1)
        score = torch.matmul(Q, K_T) / (d_head ** 0.5)

        if mask is not None:
            assert mask.size() == (batch_size, 1, 1, seq_len), \
                f"Expected mask shape: {(batch_size, 1, 1, seq_len)}, but got {mask.size()}"
            score = score.masked_fill(mask==0, float('-inf'))

        score = self.softmax(score)

        out = torch.matmul(score, V)

        return out, score

class PositionwiseFeedForward(nn.Module):
    """
    "position-wise" because the FC layers are applied along the last d_model dimension
    """
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionalEncoding, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, attention_mask):
        # Residual connection before self-attention
        residual = x  
        x = self.attention(q=x, k=x, v=x, mask=attention_mask)
        
        # Add & Norm after attention
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        
        # Residual connection before feedforward network
        residual = x  
        x = self.ffn(x)
      
        # Add & Norm after feedforward network
        x = self.dropout2(x)
        x = self.norm2(x + residual)

        return x

class Encoder(nn.Module):
    def __init__():
        pass



class Transformer:
    def __init__(self,
                 config=None) -> None:
        self.config = {
            'channels': 1,
            'image_size': 28,   # Default for MNIST
            'latent_dim': 256,
            'hidden_sizes': [512, 512],
            'encoder_hidden_channels': [64, 128],    
            'decoder_hidden_channels': [128, 64],    
            'activation': nn.ReLU()
        }    
        self.encoder = None
        self.decoder = None
        self.positional_encoding = PositionalEncoding()

    