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
    def __init__(self, input_size, tot_num_experts, noise_std: float = 0.1):
        """
        Initialize a Router module.

        Args:
            input_size (int): Dimensionality of the input to the router.
            tot_num_experts (int): Total number of experts in the mixture.
            noise_std (float): Standard deviation of Gaussian noise added during training for exploration.
        """
        super().__init__()
        self.tot_num_experts = tot_num_experts
        self.router_weights = nn.Linear(input_size, tot_num_experts, bias=False)
        self.noise_std = noise_std

    def forward(self, inputs, training: bool = False):
        """
        Forward pass of the Router module.

        Args:
            inputs (torch.Tensor): Input tensor.
            training (bool): Whether the model is in training mode or not.

        Returns:
            torch.Tensor: Routing probabilities over the experts.
        """
        routing_logits = self.router_weights(inputs)
        if training:
            routing_logits = routing_logits + torch.randn_like(routing_logits) * self.noise_std
        routing_probs = F.softmax(routing_logits, dim=-1)
        return routing_probs


class MoELayer(nn.Module):
    def __init__(self, model_dim, expert_hidden_dim, tot_num_experts, chosen_num_experts, noise_std):
        """
        Initialize a Mixture of Experts (MoE) Layer module.

        Args:
            model_dim (int): Dimensionality of the input to the MoE layer.
            expert_hidden_dim (int): Dimensionality of the hidden layer within each expert.
            tot_num_experts (int): Total number of experts in the mixture.
            chosen_num_experts (int): Number of experts to use for each input.
            noise_std (float): Standard deviation of Gaussian noise added during training for exploration.
        """
        super().__init__()
        self.model_dim = model_dim
        self.tot_num_experts = tot_num_experts
        self.chosen_num_experts = chosen_num_experts
        self.experts = nn.ModuleList([Expert(model_dim, expert_hidden_dim) for _ in range(tot_num_experts)])
        self.router = Router(model_dim, tot_num_experts, noise_std)

    def forward(self, inputs, training: bool = False):
        """
        Forward pass of the MoE Layer module.

        Args:
            inputs (torch.Tensor): Input tensor.
            training (bool): Whether the model is in training mode or not.

        Returns:
            torch.Tensor: MoE output tensor.
            torch.Tensor: Routing probabilities over the experts.
        """
        b, seq_len, _ = inputs.shape

        # Get the output of all the experts
        expert_outputs = [expert(inputs.view(-1, self.model_dim)) for expert in self.experts]
        expert_outputs = torch.cat(expert_outputs, dim=0).view(b, seq_len, self.tot_num_experts, self.model_dim)

        # Get the output of the router and create the expert mask
        routing_probs = F.softmax(self.router(inputs), dim=-1)
        with torch.no_grad():
            expert_indices = torch.topk(routing_probs, k=self.chosen_num_experts, sorted=True).indices
            multi_hot_indices = torch.zeros(b, seq_len, self.tot_num_experts, device=inputs.device)
            multi_hot_indices = multi_hot_indices.scatter(2, expert_indices, 1)

        # Apply the multi-hot mask (first expand dimensions for broadcasting)
        multi_hot_expanded = multi_hot_indices.unsqueeze(-1).expand_as(expert_outputs)
        output_masked = expert_outputs * multi_hot_expanded.float()

        # Weight our experts' outputs by the softmax values (which we first must broadcast to the right shape) and sum them
        routing_probs_expanded = routing_probs.unsqueeze(-1).expand_as(output_masked)
        MoE_output = (output_masked * routing_probs_expanded).sum(dim=2)

        return MoE_output, routing_probs  # Also output routing_probs to be used in the loss function later

