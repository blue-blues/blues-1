import torch
import torch.nn as nn
import math
import torch.nn.functional as F


from parameter import *
class RotaryEmbedding(nn.Module):
    """Rotary Embedding layer"""
    def __init__(self, dim, max_sequence_length=block_size):
        super().__init__()
        self.dim = dim
        self.max_sequence_length = max_sequence_length
        self.frequency = torch.arange(0, dim, 2) / dim

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len).to(x.device)
        freqs = self.frequency.to(x.device)
        
        # Apply rotary embedding
        emb = torch.zeros((seq_len, self.dim)).to(x.device)
        emb[:, 0::2] = torch.sin(2 * math.pi * t[:, None] * freqs[None, :])
        emb[:, 1::2] = torch.cos(2 * math.pi * t[:, None] * freqs[None, :])
        
        return emb


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_sequence_length=block_size):
        super().__init__()
        self.rotary_embedding = RotaryEmbedding(dim)
        self.positional_embedding = nn.Embedding(max_sequence_length, dim)

    def forward(self, x):
        seq_len = x.shape[1]
        rotary_emb = self.rotary_embedding(x)
        positional_emb = self.positional_embedding(torch.arange(seq_len).to(x.device))
        return rotary_emb + positional_emb
    

class MoeLayer(nn.Module):
    def __init__(self, experts, gate, k=1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.k = k

    def forward(self, inputs: torch.Tensor):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.gate(inputs_squashed)
        weights, selected_experts = torch.topk(
            gate_logits, self.k
        )
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(inputs)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs_squashed[batch_idx]
            )
        return results.view_as(inputs)



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MulitHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x =  torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(x))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4* n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
         nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)
class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts=4):
        super().__init__()
        self.rotary_pos_embedding = RotaryPositionalEmbedding(n_embed)

        self.sa_head = MulitHeadAttention(n_head, n_embed//n_head)
        self.ffw = MoeLayer(
            experts=[FeedForward(n_embed) for _ in range(num_experts)],
            gate=nn.Linear(n_embed, num_experts, bias=False),
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        rotary_pos_emb = self.rotary_pos_embedding(x)
        x = x + self.sa_head(self.ln1(x + rotary_pos_emb))
        x = x + self.ffw(self.ln2(x))
        return x


class SparseMoELanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.RotaryPositionalEmbedding = nn.Embedding(vocab_size, n_embed, device=device)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        x = token_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokes):
        for _ in range(max_new_tokes):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


# Initialize model and optimizer
model = SparseMoELanguageModel()
model.  apply(_init_weights)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Evaluation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out