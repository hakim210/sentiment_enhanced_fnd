import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

class SentimentGuidedAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_embedding: torch.Tensor, emotion_weight: float) -> torch.Tensor:
        batch_size, seq_len, _ = text_embedding.shape
        q, k, v = self.in_proj(text_embedding).chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        fused_attention_scores = attention_scores * emotion_weight
        
        attention_probs = F.softmax(fused_attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, v)
        
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context_layer)
        return output


