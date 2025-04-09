import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, C = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        queries = queries.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("bhqd,bhkd->bhqk", [queries, keys])
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)
        out = torch.einsum("bhqk,bhvd->bhqd", [attention, values])
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.fc_out(out)
