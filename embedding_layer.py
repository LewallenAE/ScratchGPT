import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len=100):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embedding_dim))

    def forward(self, x):
        return self.token_embed(x) + self.pos_embed[:, :x.size(1), :]
