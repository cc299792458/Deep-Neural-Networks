"""
    Token Embedding for Transformer
"""

import torch
import torch.nn as nn

# Define TokenEmbedding class
class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn.Embedding
    Converts token indices into dense embeddings
    """

    def __init__(self, vocab_size, d_model):
        """
        :param vocab_size: size of the vocabulary
        :param d_model: dimension of the model (embedding size)
        """
        # Initialize nn.Embedding with vocab_size, embedding dimension (d_model), and padding index
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

if __name__ == '__main__':
    # Vocabulary
    vocab = {
        '<PAD>': 1,    # Padding token
        'hello': 2,
        'world': 3,
        'how': 4,
        'are': 5,
        'you': 6
    }
    vocab_size = len(vocab) + 1  # Add 1 to account for <UNK> or other tokens
    d_model = 8  # Embedding dimension

    # Example sentence represented as token indices: ['hello', 'world', 'how', 'are', 'you']
    sample_sentence = [2, 3, 4, 5, 6]  # Corresponds to the token indices for the sentence

    # Create a batch of sentences (with padding)
    batch_size = 2
    padded_sequence = [
        [2, 3, 4, 5, 6],  # First sentence
        [2, 3, 1, 1, 1]   # Second sentence (padded with <PAD> tokens)
    ]
    padded_sequence = torch.tensor(padded_sequence)  # Convert to tensor

    # Initialize TokenEmbedding layer
    embedding_layer = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)

    # Get embeddings for the input token indices
    embeddings = embedding_layer(padded_sequence)

    # Print results
    print("Shape of embedded tensor: ", embeddings.shape)  # Output: [batch_size, seq_len, d_model]
    print("Embedding vectors: \n", embeddings)
        