import math
import torch
from torch import nn
from torch.nn import functional as F
from configuration import Config


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

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        q = q.reshape(q.size(0), q.size(1), self.num_heads, self.head_dim)
        k = k.reshape(k.size(0), k.size(1), self.num_heads, self.head_dim)
        v = v.reshape(v.size(0), v.size(1), self.num_heads, self.head_dim)
        output = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        output = self.dropout(F.softmax(output, dim=-1))
        output = torch.matmul(output, v)
        output = output.reshape(q.size(0), q.size(1), -1)
        output = self.Wo(output)
        return output

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

    def forward(self, x):
        y = self.attention(x)
        x = self.add_norm1(x, y)
        y = self.mlp(x)
        y = self.add_norm2(x, y)
        return y

class TransformerDecoder(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(config.vocab_size, config.num_hiddens)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.post_ln = nn.LayerNorm(config.num_hiddens)

    def forward(self, x: torch.LongTensor):
        hidden_states = self.embedding(x)
        for idx, decoder_block in enumerate(self.decoder_blocks):
            hidden_states = decoder_block(hidden_states)
        hidden_states = self.post_ln(hidden_states)
        return hidden_states

class CasualLM(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.decoder = TransformerDecoder(config)
        self.lm_head = nn.Linear(config.num_hiddens, config.vocab_size)

    def forward(self, x: torch.LongTensor):
        hidden_states = self.decoder(x)
        logits = self.lm_head(hidden_states)
        return logits
