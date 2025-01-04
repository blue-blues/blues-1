import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from moe import *
from config import config

class blues(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0
        self.max_seq_len = config.max_position_embeddings
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.tokenizer = tokenizer
        self.embedder = nn.Embedding(self.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.criterion = nn.CrossEntropyLoss()
        self.lambadada = config.lambadada

    def calc_moe_loss(self, routing_probs_list):
        all_routing_probs = torch.cat([x.unsqueeze(0) for x in routing_probs_list], dim=0)
        expert_usage = all_routing_probs.sum(dim=(1, 2))
        usage_mean = expert_usage.mean(dim=0)
        expert_variance = ((expert_usage - usage_mean) ** 2).mean(dim=0)
        cum_var = expert_variance.sum()
        return cum_var

    def forward(self, input_token_ids: torch.Tensor, target_token_ids: torch.Tensor = None) -> torch.Tensor:
        training = target_token_ids is not None
        x = self.embedder(input_token_ids) * self.config.hidden_size ** 0.5
        routing_probs_list = []
        for layer in self.layers:
            x, routing_probs = layer(x, training)
            if training:
                routing_probs_list.append(routing_probs)
        x = self.final_norm(x)
        logits = torch.matmul(x, self.embedder.weight.t())
        if training:
            batch_size, input_len, vocab_size = logits.shape
            CEloss = self.criterion(logits.view(batch_size * input_len, vocab_size), target_token_ids.view(batch_size * input_len))
            MoEloss = self.calc_moe_loss(routing_probs_list)
            loss = CEloss + MoEloss * self.lambadada
        else:
            loss = None
        return logits, loss

    @torch.no_grad()
    def Sampler(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
        logits = logits[:, -1, :]
        logits.div_(temperature)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_p
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device).expand(probs_idx.shape[0], -1) >= top_k
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
        next_token_id = torch.multinomial(probs, num_samples=1)
        return next_token_id

    def generate(self, prompt: str, output_len: int = 100, temperature: float = 0.95, top_p: float = 1.0, top_k: int = 65) -> str:
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, device=self.config.device).unsqueeze(0)
        assert len(tokens) + output_len <= self.config.max_position_embeddings
        for _ in range(output_len):
            logits, _ = self(tokens[:, :self.max_seq_len])
            next_token = self.Sampler(logits=logits, temperature=temperature, top_p=top_p, top_k=top_k)
            tokens = torch.cat((tokens, next_token), dim=1)
        output = self.tokenizer.decode(tokens.squeeze(0).tolist())
        return output

class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, use_scale=True):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features)) if use_scale else None

    def forward(self, x):
        mean_squared = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_squared + self.eps)
        if self.scale is not None:
            x = x * self.scale
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mqa = MQA(config)
        self.moe = MoELayer(
            model_dim=config.hidden_size,
            expert_hidden_dim=config.hidden_size * config.embedding_multiplier_scale,
            tot_num_experts=config.tot_num_experts,
            chosen_num_experts=config.chosen_num_experts,
            noise_std=config.noise_std
        )
        self.pre_mqa_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_scale=config.use_scale)
        self.post_mqa_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_scale=config.use_scale)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_scale=config.use_scale)
        self.post_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_scale=config.use_scale)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if training:
            x = x + self.drop(self.post_mqa_norm(self.mqa(self.pre_mqa_norm(x))))
            moe_out, routing_probs = self.moe(self.pre_moe_norm(x), training)
            x = x + self.drop(self.post_moe_norm(moe_out))
        else:
            x = x + self.post_mqa_norm(self.mqa(self.pre_mqa_norm(x)))
            moe_out, routing_probs = self.moe(self.pre_moe_norm(x), training)
            x = x + self.post_moe_norm(moe_out)
        return x, routing_probs

@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters=10):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
