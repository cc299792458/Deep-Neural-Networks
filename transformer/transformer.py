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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head) -> None:
        super(MultiHeadAttention, self).__init__()

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
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
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
    def __init__(self, voc_size, max_len, d_model, ffn_hidden, n_head, n_blocks, drop_prob, device):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbedding(vocab_size=voc_size, 
                                        d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        self.blocks = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                                  for _ in range(n_blocks)])
    
    def forward(self, x, attention_mask):
        x = self.emb(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderBlock, self).__init__()
        # Self-attention
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # Cross-attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # Feed-forward network
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, encoder_output, x_mask, encoder_mask):
        # 1. Self-attention
        residual = x
        x = self.self_attention(q=x, k=x, v=x, mask=x_mask)
        
        # 2. Add & Norm
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        if encoder_output is not None:
            # 3. Cross-attention (with encoder output)
            residual = x
            x = self.cross_attention(q=x, k=encoder_output, v=encoder_output, mask=encoder_mask)
            
            # 4. Add & Norm
            x = self.dropout2(x)
            x = self.norm2(x + residual)

        # 5. Position-wise feed-forward network
        residual = x
        x = self.ffn(x)
        
        # 6. Add & Norm
        x = self.dropout3(x)
        x = self.norm3(x + residual)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, ffn_hidden, n_head, n_blocks, drop_prob, device):
        super().__init__()
        # Embedding layer with positional encoding
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=vocab_size,
                                        device=device)

        # Stack of decoder layers
        self.layers = nn.ModuleList([DecoderBlock(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                                  for _ in range(n_blocks)])

        # Linear projection to vocabulary size (Language Model Head)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, x_mask, encoder_mask):
        # Apply embedding and positional encoding
        x = self.emb(x)

        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, x_mask, encoder_mask)

        # Linear projection to vocabulary logits
        output = self.linear(x)

        return output

class Transformer:
    def __init__(self, config=None) -> None:
        # Set default config and update with any user-provided values
        self.config = self.default_config()
        if config is not None:
            self.config.update(config)
        
        # Initialize components with parameters from the config
        self.encoder = Encoder(
            vocab_size=self.config['src_vocab_size'],
            d_model=self.config['d_model'],
            n_head=self.config['n_head'],
            max_len=self.config['max_len'],
            ffn_hidden=self.config['ffn_hidden'],
            n_blocks=self.config['n_blocks'],
            drop_prob=self.config['drop_prob'],
            device=self.config['device']
        )
        
        self.decoder = Decoder(
            vocab_size=self.config['tgt_vocab_size'],
            d_model=self.config['d_model'],
            n_head=self.config['n_head'],
            max_len=self.config['max_len'],
            ffn_hidden=self.config['ffn_hidden'],
            n_blocks=self.config['n_blocks'],
            drop_prob=self.config['drop_prob'],
            device=self.config['device']
        )
    
    def default_config(self):
        """ Default configuration for Transformer """
        return {
            'src_vocab_size': 10000,
            'tgt_vocab_size': 10000,
            'd_model': 512,
            'n_head': 8,
            'max_len': 256,
            'ffn_hidden': 2048,
            'n_blocks': 6,
            'drop_prob': 0.1,
            'src_pad_idx': 0,
            'tgt_pad_idx': 0,
            'tgt_sos_idx': 1,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    def forward(self, input_seq, target_seq):
        input_mask = self.make_input_mask(input_seq)
        target_mask = self.make_target_mask(target_seq)
        enc_input = self.encoder(input_seq, input_mask)
        output = self.decoder(target_seq, enc_input, target_mask, input_mask)
        return output

    def make_input_mask(self, input_seq):
        input_mask = (input_seq != self.config['src_pad_idx']).unsqueeze(1).unsqueeze(2)
        return input_mask

    def make_target_mask(self, target_seq):
        target_pad_mask = (target_seq != self.config['tgt_pad_idx']).unsqueeze(1).unsqueeze(3)
        target_len = target_seq.shape[1]
        target_sub_mask = torch.tril(torch.ones((target_len, target_len))).type(torch.ByteTensor).to(self.config['device'])
        target_mask = target_pad_mask & target_sub_mask
        return target_mask

    