import math
import torch
from torch import nn
from torch.nn import functional as F
from configuration import Config


class PositionEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.P = torch.zeros((1, config.max_context_length, config.num_hiddens))
        X = torch.arange(config.max_context_length, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, config.num_hiddens, 2, dtype=torch.float32) / config.num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)

class AddNorm(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.num_hiddens)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = config.num_heads
        self.head_dim = int(config.num_hiddens / config.num_heads)
        self.Wq = nn.Linear(config.num_hiddens, config.num_hiddens, bias=False)
        self.Wk = nn.Linear(config.num_hiddens, config.num_hiddens, bias=False)
        self.Wv = nn.Linear(config.num_hiddens, config.num_hiddens, bias=False)
        self.Wo = nn.Linear(config.num_hiddens, config.num_hiddens, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, past_key_value=None, use_cache: bool=False):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        q = q.reshape(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        if use_cache:
            past_key_value = (k, v)
        output = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        output = self.dropout(F.softmax(output, dim=-1))
        output = torch.matmul(output, v).transpose(1, 2).contiguous()
        output = output.reshape(q.size(0), q.size(1), -1)
        output = self.Wo(output)
        return output, past_key_value

class MLP(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.linear1 = nn.Linear(config.num_hiddens, config.num_mlp_intermediate)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.num_mlp_intermediate, config.num_hiddens)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class DecoderBlock(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadSelfAttention(config)
        self.add_norm1 = AddNorm(config)
        self.mlp = MLP(config)
        self.add_norm2 = AddNorm(config)

    def forward(self, x, past_key_value=None, use_cache: bool=False):
        y, past_key_value = self.attention(x, past_key_value, use_cache)
        x = self.add_norm1(x, y)
        y = self.mlp(x)
        y = self.add_norm2(x, y)
        return y, past_key_value

class TransformerDecoder(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = config.num_layers
        self.embedding = nn.Embedding(config.vocab_size, config.num_hiddens)
        self.pos_embedding = PositionEmbedding(config)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.post_ln = nn.LayerNorm(config.num_hiddens)

    def forward(self, x: torch.LongTensor, past_key_values=None, use_cache: bool=False):
        embedding = self.embedding(x)
        hidden_states = self.pos_embedding(embedding)
        if use_cache and past_key_values is None:
            past_key_values = [None] * self.num_layers
        for idx, decoder_block in enumerate(self.decoder_blocks):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            hidden_states, past_key_value = decoder_block(hidden_states, past_key_value, use_cache)
            if past_key_values is not None:
                past_key_values[idx] = past_key_value
        hidden_states = self.post_ln(hidden_states)
        return hidden_states, past_key_values

class CasualLM(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.decoder = TransformerDecoder(config)
        self.lm_head = nn.Linear(config.num_hiddens, config.vocab_size)

    def forward(self, x: torch.LongTensor, past_key_values=None, use_cache: bool=False):
        hidden_states, past_key_values = self.decoder(x, past_key_values, use_cache)
        logits = self.lm_head(hidden_states)
        return logits, past_key_values
    
    def generate(self, x: torch.LongTensor, max_new_tokens: int=10):
        pass
