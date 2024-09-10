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
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_encoding(x)
        return self.dropout(token_emb + pos_emb)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention = ScaledDotProductAttention()

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        query, key, value = self.query_linear(query), self.key_linear(key), self.value_linear(value)
        query, key, value = self.split_heads(query), self.split_heads(key), self.split_heads(value)
        attention_out, attention_weights = self.attention(query, key, value, mask)
        attention_out = self.concat_heads(attention_out)
        return self.output_linear(attention_out)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        d_head = d_model // self.n_heads
        return x.view(batch_size, seq_len, self.n_heads, d_head).transpose(1, 2)

    def concat_heads(self, x):
        batch_size, n_heads, seq_len, d_head = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, n_heads * d_head)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size, n_heads, seq_len, d_head = key.size()
        key_T = key.transpose(-2, -1)
        scores = torch.matmul(query, key_T) / (d_head ** 0.5)

        if mask is not None:
            assert mask.size() == (batch_size, 1, 1, seq_len), \
                f"Expected mask shape: {(batch_size, 1, 1, seq_len)}, but got {mask.size()}"
            scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = self.softmax(scores)
        return torch.matmul(scores, value), scores

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden_dim, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden_dim)
        self.linear2 = nn.Linear(ffn_hidden_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden_dim, n_heads, drop_prob):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_hidden_dim=ffn_hidden_dim, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask):
        residual = x  
        x = self.self_attention(query=x, key=x, value=x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        residual = x  
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, ffn_hidden_dim, n_heads, n_blocks, drop_prob, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len,
                                              drop_prob=drop_prob, device=device)
        self.blocks = nn.ModuleList([EncoderBlock(d_model=d_model, ffn_hidden_dim=ffn_hidden_dim, n_heads=n_heads,
                                                  drop_prob=drop_prob) for _ in range(n_blocks)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for blocks in self.blocks:
            x = blocks(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden_dim, n_heads, drop_prob):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.cross_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_hidden_dim=ffn_hidden_dim, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, encoder_output, self_mask, cross_mask):
        residual = x
        x = self.self_attention(query=x, key=x, value=x, mask=self_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        residual = x
        x = self.cross_attention(query=x, key=encoder_output, value=encoder_output, mask=cross_mask)
        x = self.dropout2(x)
        x = self.norm2(x + residual)

        residual = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + residual)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, ffn_hidden_dim, n_heads, n_blocks, drop_prob, device):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len,
                                              drop_prob=drop_prob, device=device)
        self.blocks = nn.ModuleList([DecoderBlock(d_model=d_model, ffn_hidden_dim=ffn_hidden_dim, n_heads=n_heads,
                                                  drop_prob=drop_prob) for _ in range(n_blocks)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, self_mask, cross_mask):
        x = self.embedding(x)
        for blocks in self.blocks:
            x = blocks(x, encoder_output, self_mask, cross_mask)
        return self.linear(x)

class Transformer(nn.Module):
    def __init__(self, config=None):
        super(Transformer, self).__init__()
        self.config = self.default_config()
        if config:
            self.config.update(config)

        self.encoder = Encoder(
            vocab_size=self.config['src_vocab_size'],
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            max_len=self.config['max_len'],
            ffn_hidden_dim=self.config['ffn_hidden_dim'],
            n_blocks=self.config['n_blocks'],
            drop_prob=self.config['drop_prob'],
            device=self.config['device']
        )

        self.decoder = Decoder(
            vocab_size=self.config['tgt_vocab_size'],
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            max_len=self.config['max_len'],
            ffn_hidden_dim=self.config['ffn_hidden_dim'],
            n_blocks=self.config['n_blocks'],
            drop_prob=self.config['drop_prob'],
            device=self.config['device']
        )

    def default_config(self):
        return {
            'src_vocab_size': 10000,
            'tgt_vocab_size': 10000,
            'd_model': 512,
            'n_heads': 8,
            'max_len': 256,
            'ffn_hidden_dim': 2048,
            'n_blocks': 6,
            'drop_prob': 0.1,
            'src_pad_idx': 0,
            'tgt_pad_idx': 0,
            'tgt_sos_idx': 1,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.config['src_pad_idx']).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != self.config['tgt_pad_idx']).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len))).type(torch.ByteTensor).to(self.config['device'])
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
