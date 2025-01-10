import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Conditional import for flash_attn
try:
    from flash_attn import flash_attn_func, flash_attn_kvpacked_func
    flash_attn_available = True
except ImportError:
    flash_attn_available = False

def apply_rotary_emb(x: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    seq_len = x.size(1)
    device = x.device
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis.unsqueeze(0)).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
    return x_out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v), attn

class FlashMHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, q, k, v, key_padding_mask=None, causal=True):
        """
        q, k, v: (batch_size, seqlen, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen)
        """
        batch_size, seqlen = q.shape[0], q.shape[1]
        
        # Reshape and prepare inputs for flash attention
        q = q.reshape(batch_size, seqlen, -1)
        k = k.reshape(batch_size, seqlen, -1)
        v = v.reshape(batch_size, seqlen, -1)
        
        # Run flash attention
        if flash_attn_available:
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=causal,
                softmax_scale=self.softmax_scale
            )
        else:
            # Fallback to standard attention if flash_attn is not available
            attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.softmax_scale, dim=-1)
            out = torch.matmul(attn, v)
        
        return out

class MQA(nn.Module):
    """
    Implements Multi-Query Attention which supports a distinct number of attention heads for queries and key-values (KV).
    In the case where the same number of queries and key-values are used, this implementation is equivalent to regular Multi-Head Attention.
    """
    def __init__(self, config):
        """
        Initializes the MQA module.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.theta = config.rope_theta

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Projection layers
        self.qkv_proj = nn.Linear(self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Register mask as a buffer for proper device handling
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    (config.max_position_embeddings, config.max_position_embeddings),
                    dtype=torch.bool
                )
            ).view(1, 1, config.max_position_embeddings, config.max_position_embeddings)
        )

        # Conditionally initialize attention mechanism
        if torch.cuda.is_available() and config.use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self.flash_attention = True
                print("Using Flash Attention")
            except ImportError:
                self.flash_attention = False
                print("Flash Attention not available, using standard attention")
        else:
            self.flash_attention = False
            print("Using standard attention (CPU or Flash Attention disabled)")
        
        if self.flash_attention:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
        else:
            self.attention = ScaledDotProductAttention(dropout=config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MQA module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Output tensor after Multi-Query Attention mechanism.
        """
        batch_size, input_len, _ = hidden_states.shape

        # Linear projection to retrieve q, k, v projections
        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape to separate heads and align dimensions for attention operations
        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Apply rotary positional embeddings to queries and keys
        xq = apply_rotary_emb(xq, self.head_dim, self.theta)
        xk = apply_rotary_emb(xk, self.head_dim, self.theta)

        # Adjust keys and values if the number of KV heads differs from the number of query heads
        if self.num_kv_heads != self.num_heads:
            xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv, self.num_queries_per_kv, dim=2)

        # Transpose to align for batch matrix multiplication in attention calculation
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        if self.flash_attention:
            # Reshape for flash attention
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            output = self.flash_attn_func(q, k, v, softmax_scale=1.0/math.sqrt(self.head_dim))
        else:
            # Use standard attention
            output, _ = self.attention(q, k, v, self.mask[:, :, :input_len, :input_len])

        # Reshape attention output to combine heads back into hidden dimension
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)

        # Final linear projection to map back to hidden size dimension
        output = self.o_proj(output)

        return output
        

class Expert(nn.Module):
    def __init__(self, model_dim, hidden_dim):
        """
        Initialize an Expert module.

        Args:
            model_dim (int): Dimensionality of the input to the expert.
            hidden_dim (int): Dimensionality of the hidden layer within the expert.
        """
        super().__init__()
        self.layer1 = nn.Linear(model_dim, hidden_dim * 2, bias=False)  # Double the output for gating
        self.layer2 = nn.Linear(hidden_dim, model_dim, bias=False)  # Output layer remains the same

    def forward(self, x):
        """
        Forward pass of the Expert module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the Expert network.
        """
        # Split the output of the first layer for gating
        x, gate = self.layer1(x).chunk(2, dim=-1)

        # Apply GeLU to the gate, and then multiply element-wise
        x = F.gelu(gate) * x
        x = self.layer2(x)

        return x


class Router(nn.Module):
    def __init__(self, input_size, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_size, num_experts, bias=False)
        
    def forward(self, x):
        # Calculate routing weights
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        weights = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Create routing mask
        mask = torch.zeros_like(weights).scatter_(-1, top_k_indices, top_k_weights)
        
        return mask, top_k_indices

class ExpertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.expert_ffn_size)
        self.fc2 = nn.Linear(config.expert_ffn_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

from expert_pruning import DynamicExpertPruning

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use consistent configuration parameters
        self.num_experts = config.num_experts
        self.input_size = config.hidden_size  # Use hidden_size instead of n_embd
        self.hidden_size = config.hidden_size
        self.expert_hidden_size = config.expert_ffn_size
        self.top_k = config.top_k
        
        # Router
        self.router = Router(self.input_size, self.num_experts, self.top_k)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(self.hidden_size, self.expert_hidden_size) 
            for _ in range(self.num_experts)
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # Initialize expert pruning with fallback values
        enable_pruning = getattr(config, 'enable_expert_pruning', False)
        self.expert_pruning = DynamicExpertPruning(config) if enable_pruning else None
        self.training_step = 0

    def forward(self, x, training=False):  # Add training parameter
        identity = x
        x = self.layer_norm(x)
        
        # Get routing weights and expert assignments
        router_mask, expert_indices = self.router(x)
        
        # Process input through experts
        expert_outputs = torch.zeros_like(x)
        for i in range(self.num_experts):
            expert_mask = router_mask[:, :, i].unsqueeze(-1)
            if expert_mask.any():
                # Skip pruned experts
                if self.expert_pruning and i in self.expert_pruning.state.pruned_experts:
                    continue
                expert_output = self.experts[i](x)
                expert_outputs += expert_output * expert_mask
        
        # Update expert pruning statistics
        if self.expert_pruning is not None and training:
            self.expert_pruning.update(expert_indices, router_mask)
            self.expert_pruning.apply_to_parameters(self.experts)
            self.training_step += 1
        
        output = self.dropout(expert_outputs)
        return output + identity, router_mask

    def get_pruning_stats(self):
        """Get expert pruning statistics"""
        if self.expert_pruning is not None:
            return self.expert_pruning.get_expert_stats()
        return None

    def save_pruning_state(self, checkpoint):
        """Save pruning state to checkpoint"""
        if self.expert_pruning is not None:
            checkpoint['pruning_state'] = self.expert_pruning.get_state_dict()

    def load_pruning_state(self, checkpoint):
        """Load pruning state from checkpoint"""
        if self.expert_pruning is not None and 'pruning_state' in checkpoint:
            self.expert_pruning.load_state_dict(checkpoint['pruning_state'])

def load_balancing_loss(router_mask):
    """Calculate load balancing loss for expert routing"""
    # Calculate expert usage
    expert_usage = router_mask.sum(dim=[0, 1])  # Sum over batch and sequence
    # Ideal balanced load
    ideal_load = torch.ones_like(expert_usage) * router_mask.sum() / len(expert_usage)
    # Calculate loss
    loss = torch.mean((expert_usage - ideal_load) ** 2)
    return loss

