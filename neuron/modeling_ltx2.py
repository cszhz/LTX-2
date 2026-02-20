# coding=utf-8
"""LTX-2 Video Transformer for Neuron inference with Tensor Parallelism."""

import math
import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)

from config import LTX2NeuronConfig

logger = logging.getLogger("Neuron-LTX2")


def get_activation_fn(name: str):
    """Get activation function by name."""
    if name == "gelu-approximate":
        return lambda x: F.gelu(x, approximate="tanh")
    elif name == "gelu":
        return F.gelu
    elif name == "silu":
        return F.silu
    else:
        raise ValueError(f"Unknown activation: {name}")


class RMSNorm(nn.Module):
    """RMS Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class LTX2RoPE3D(nn.Module):
    """3D Rotary Position Embedding for LTX-2 (split type)."""

    def __init__(
        self,
        dim: int,
        max_temporal_pos: int = 20,
        max_height: int = 2048,
        max_width: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_temporal_pos = max_temporal_pos
        self.max_height = max_height
        self.max_width = max_width
        self.theta = theta

        # Split dimensions for T, H, W
        self.dim_t = dim // 4
        self.dim_h = dim // 4
        self.dim_w = dim // 2

        # Precompute inverse frequencies
        inv_freq_t = 1.0 / (theta ** (torch.arange(0, self.dim_t, 2).float() / self.dim_t))
        inv_freq_h = 1.0 / (theta ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
        inv_freq_w = 1.0 / (theta ** (torch.arange(0, self.dim_w, 2).float() / self.dim_w))

        self.register_buffer("inv_freq_t", inv_freq_t, persistent=False)
        self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        positions_t: torch.Tensor,
        positions_h: torch.Tensor,
        positions_w: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 3D RoPE.

        Args:
            x: [B, N, H, D] or [B, N, D]
            positions_t: [B, N] temporal positions
            positions_h: [B, N] height positions
            positions_w: [B, N] width positions
        """
        # Compute frequencies
        freqs_t = torch.einsum("bn,d->bnd", positions_t.float(), self.inv_freq_t)
        freqs_h = torch.einsum("bn,d->bnd", positions_h.float(), self.inv_freq_h)
        freqs_w = torch.einsum("bn,d->bnd", positions_w.float(), self.inv_freq_w)

        # Concatenate and create cos/sin
        freqs = torch.cat([freqs_t, freqs_t, freqs_h, freqs_h, freqs_w, freqs_w], dim=-1)
        cos = freqs.cos()
        sin = freqs.sin()

        # Handle multi-head case
        if x.dim() == 4:
            cos = cos.unsqueeze(2)  # [B, N, 1, D]
            sin = sin.unsqueeze(2)

        # Apply rotation
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos1 = cos[..., ::2]
        sin1 = sin[..., ::2]

        rotated = torch.stack([
            x1 * cos1 - x2 * sin1,
            x1 * sin1 + x2 * cos1,
        ], dim=-1).flatten(-2)

        return rotated


class NeuronLTX2Attention(nn.Module):
    """LTX-2 Attention with Tensor Parallelism."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        num_heads: int = 32,
        head_dim: int = 128,
        bias: bool = True,
        tp_degree: int = 4,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim or query_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.tp_degree = tp_degree

        # Heads per TP rank
        assert num_heads % tp_degree == 0, f"num_heads ({num_heads}) must be divisible by tp_degree ({tp_degree})"
        self.num_heads_per_partition = num_heads // tp_degree
        self.inner_dim_per_partition = self.num_heads_per_partition * head_dim

        if parallel_state.model_parallel_is_initialized():
            # Q projection
            self.to_q = ColumnParallelLinear(
                query_dim,
                self.inner_dim,
                bias=bias,
                gather_output=False,
                dtype=dtype,
            )
            # K projection
            self.to_k = ColumnParallelLinear(
                self.cross_attention_dim,
                self.inner_dim,
                bias=bias,
                gather_output=False,
                dtype=dtype,
            )
            # V projection
            self.to_v = ColumnParallelLinear(
                self.cross_attention_dim,
                self.inner_dim,
                bias=bias,
                gather_output=False,
                dtype=dtype,
            )
            # Output projection
            self.to_out = RowParallelLinear(
                self.inner_dim,
                query_dim,
                bias=bias,
                input_is_parallel=True,
                dtype=dtype,
            )
        else:
            self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_out = nn.Linear(self.inner_dim, query_dim, bias=bias)

        # QK normalization
        self.norm_q = RMSNorm(head_dim)
        self.norm_k = RMSNorm(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rope_embeds: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, N, D]
            encoder_hidden_states: [B, M, D] for cross-attention
            attention_mask: [B, N, M] or [B, 1, N, M]
            rope_embeds: (positions_t, positions_h, positions_w) for RoPE
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Cross-attention or self-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Project Q, K, V
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        # After TP: [B, N, num_heads_per_partition * head_dim]
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads_per_partition, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads_per_partition, self.head_dim)

        # QK normalization
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Transpose for attention: [B, H, N, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [B, H, N, D] -> [B, N, H*D]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.to_out(attn_output)

        return output


class NeuronLTX2FeedForward(nn.Module):
    """LTX-2 FeedForward with Tensor Parallelism."""

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        activation_fn: str = "gelu-approximate",
        tp_degree: int = 4,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.activation = get_activation_fn(activation_fn)

        if parallel_state.model_parallel_is_initialized():
            # GEGLU style: project to 2x inner_dim, split, multiply
            self.proj = ColumnParallelLinear(
                dim,
                inner_dim * 2,  # For GEGLU gate
                bias=True,
                gather_output=False,
                dtype=dtype,
            )
            self.out_proj = RowParallelLinear(
                inner_dim,
                dim,
                bias=True,
                input_is_parallel=True,
                dtype=dtype,
            )
        else:
            self.proj = nn.Linear(dim, inner_dim * 2, bias=True)
            self.out_proj = nn.Linear(inner_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GEGLU activation
        hidden = self.proj(x)
        hidden, gate = hidden.chunk(2, dim=-1)
        hidden = hidden * self.activation(gate)
        output = self.out_proj(hidden)
        return output


class NeuronLTX2TransformerBlock(nn.Module):
    """Single LTX-2 Transformer Block with TP support."""

    def __init__(
        self,
        config: LTX2NeuronConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Video branch
        self.attn1 = NeuronLTX2Attention(
            query_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
            tp_degree=config.tp_degree,
        )
        self.attn2 = NeuronLTX2Attention(
            query_dim=config.hidden_size,
            cross_attention_dim=config.cross_attention_dim,
            num_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
            tp_degree=config.tp_degree,
        )
        self.ff = NeuronLTX2FeedForward(
            dim=config.hidden_size,
            inner_dim=config.ff_inner_dim // 2,  # GEGLU doubles internally
            activation_fn=config.activation_fn,
            tp_degree=config.tp_degree,
        )

        # Audio branch
        self.audio_attn1 = NeuronLTX2Attention(
            query_dim=config.audio_hidden_size,
            num_heads=config.audio_num_attention_heads,
            head_dim=config.audio_attention_head_dim,
            tp_degree=config.tp_degree,
        )
        self.audio_attn2 = NeuronLTX2Attention(
            query_dim=config.audio_hidden_size,
            cross_attention_dim=config.audio_cross_attention_dim,
            num_heads=config.audio_num_attention_heads,
            head_dim=config.audio_attention_head_dim,
            tp_degree=config.tp_degree,
        )
        self.audio_ff = NeuronLTX2FeedForward(
            dim=config.audio_hidden_size,
            inner_dim=config.audio_ff_inner_dim // 2,
            activation_fn=config.activation_fn,
            tp_degree=config.tp_degree,
        )

        # Cross-modal attention
        self.audio_to_video_attn = NeuronLTX2Attention(
            query_dim=config.hidden_size,
            cross_attention_dim=config.audio_hidden_size,
            num_heads=config.audio_num_attention_heads,
            head_dim=config.audio_attention_head_dim,
            tp_degree=config.tp_degree,
        )
        self.video_to_audio_attn = NeuronLTX2Attention(
            query_dim=config.audio_hidden_size,
            cross_attention_dim=config.hidden_size,
            num_heads=config.audio_num_attention_heads,
            head_dim=config.audio_attention_head_dim,
            tp_degree=config.tp_degree,
        )

        # AdaLN modulation tables
        self.scale_shift_table = nn.Parameter(torch.randn(6, config.hidden_size) * 0.02)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(6, config.audio_hidden_size) * 0.02)
        self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(
            torch.randn(5, config.hidden_size) * 0.02
        )
        self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(
            torch.randn(5, config.audio_hidden_size) * 0.02
        )

    def forward(
        self,
        video_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        video_encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        video_temb: torch.Tensor,
        audio_temb: torch.Tensor,
        video_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for video-audio joint transformer block.

        Args:
            video_hidden_states: [B, N_v, D_v]
            audio_hidden_states: [B, N_a, D_a]
            video_encoder_hidden_states: [B, M_v, D_v] text embeddings for video
            audio_encoder_hidden_states: [B, M_a, D_a] text embeddings for audio
            video_temb: [B, 6, D_v] timestep embedding modulation
            audio_temb: [B, 6, D_a] audio timestep modulation

        Returns:
            (video_output, audio_output)
        """
        # Get modulation parameters
        video_shift_msa, video_scale_msa, video_gate_msa, video_shift_mlp, video_scale_mlp, video_gate_mlp = (
            self.scale_shift_table[None] + video_temb
        ).chunk(6, dim=1)

        audio_shift_msa, audio_scale_msa, audio_gate_msa, audio_shift_mlp, audio_scale_mlp, audio_gate_mlp = (
            self.audio_scale_shift_table[None] + audio_temb
        ).chunk(6, dim=1)

        # Video self-attention with AdaLN
        video_norm = video_hidden_states * (1 + video_scale_msa) + video_shift_msa
        video_attn_out = self.attn1(video_norm)
        video_hidden_states = video_hidden_states + video_gate_msa * video_attn_out

        # Video cross-attention with text
        video_attn_out = self.attn2(video_hidden_states, video_encoder_hidden_states)
        video_hidden_states = video_hidden_states + video_attn_out

        # Audio self-attention with AdaLN
        audio_norm = audio_hidden_states * (1 + audio_scale_msa) + audio_shift_msa
        audio_attn_out = self.audio_attn1(audio_norm)
        audio_hidden_states = audio_hidden_states + audio_gate_msa * audio_attn_out

        # Audio cross-attention with text
        audio_attn_out = self.audio_attn2(audio_hidden_states, audio_encoder_hidden_states)
        audio_hidden_states = audio_hidden_states + audio_attn_out

        # Cross-modal attention: audio -> video
        a2v_attn_out = self.audio_to_video_attn(video_hidden_states, audio_hidden_states)
        video_hidden_states = video_hidden_states + a2v_attn_out

        # Cross-modal attention: video -> audio
        v2a_attn_out = self.video_to_audio_attn(audio_hidden_states, video_hidden_states)
        audio_hidden_states = audio_hidden_states + v2a_attn_out

        # Video FFN with AdaLN
        video_norm = video_hidden_states * (1 + video_scale_mlp) + video_shift_mlp
        video_ff_out = self.ff(video_norm)
        video_hidden_states = video_hidden_states + video_gate_mlp * video_ff_out

        # Audio FFN with AdaLN
        audio_norm = audio_hidden_states * (1 + audio_scale_mlp) + audio_shift_mlp
        audio_ff_out = self.audio_ff(audio_norm)
        audio_hidden_states = audio_hidden_states + audio_gate_mlp * audio_ff_out

        return video_hidden_states, audio_hidden_states


class NeuronLTX2Transformer(nn.Module):
    """LTX-2 Video Transformer Model for Neuron."""

    def __init__(self, config: LTX2NeuronConfig):
        super().__init__()
        self.config = config

        # Timestep embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(256, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.audio_time_embed = nn.Sequential(
            nn.Linear(256, config.audio_hidden_size),
            nn.SiLU(),
            nn.Linear(config.audio_hidden_size, config.audio_hidden_size),
        )

        # Caption projections
        self.caption_projection = nn.Sequential(
            nn.Linear(config.caption_channels, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.audio_caption_projection = nn.Sequential(
            nn.Linear(config.caption_channels, config.audio_hidden_size),
            nn.SiLU(),
            nn.Linear(config.audio_hidden_size, config.audio_hidden_size),
        )

        # Input projections
        self.proj_in = nn.Linear(config.in_channels, config.hidden_size)
        self.audio_proj_in = nn.Linear(config.audio_in_channels, config.audio_hidden_size)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            NeuronLTX2TransformerBlock(config, i)
            for i in range(config.num_layers)
        ])

        # Output projections
        self.proj_out = nn.Linear(config.hidden_size, config.out_channels)
        self.audio_proj_out = nn.Linear(config.audio_hidden_size, config.audio_out_channels)

        # Global scale/shift tables for final norm
        self.scale_shift_table = nn.Parameter(torch.randn(2, config.hidden_size) * 0.02)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(2, config.audio_hidden_size) * 0.02)

    def get_timestep_embedding(self, timesteps: torch.Tensor, dim: int = 256) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            video_latents: [B, C, T, H, W] video latent
            audio_latents: [B, C_a, T_a] audio latent
            timestep: [B] diffusion timestep
            encoder_hidden_states: [B, S, D_text] text embeddings
            encoder_attention_mask: [B, S] text attention mask

        Returns:
            (video_output, audio_output) noise predictions
        """
        batch_size = video_latents.shape[0]

        # Reshape video: [B, C, T, H, W] -> [B, T*H*W, C]
        video_latents = video_latents.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, self.config.in_channels)

        # Reshape audio: [B, C, T] -> [B, T, C]
        audio_latents = audio_latents.permute(0, 2, 1)

        # Timestep embeddings
        t_emb = self.get_timestep_embedding(timestep)
        video_temb = self.time_embed(t_emb)
        audio_temb = self.audio_time_embed(t_emb)

        # Expand temb for modulation: [B, D] -> [B, 6, D]
        video_temb = video_temb.unsqueeze(1).expand(-1, 6, -1)
        audio_temb = audio_temb.unsqueeze(1).expand(-1, 6, -1)

        # Project inputs
        video_hidden = self.proj_in(video_latents)
        audio_hidden = self.audio_proj_in(audio_latents)

        # Project text embeddings
        video_encoder_states = self.caption_projection(encoder_hidden_states)
        audio_encoder_states = self.audio_caption_projection(encoder_hidden_states)

        # Transformer blocks
        for block in self.transformer_blocks:
            video_hidden, audio_hidden = block(
                video_hidden,
                audio_hidden,
                video_encoder_states,
                audio_encoder_states,
                video_temb,
                audio_temb,
            )

        # Final normalization and output projection
        video_shift, video_scale = (self.scale_shift_table[None] + video_temb[:, :2]).chunk(2, dim=1)
        video_hidden = video_hidden * (1 + video_scale.squeeze(1)) + video_shift.squeeze(1)
        video_output = self.proj_out(video_hidden)

        audio_shift, audio_scale = (self.audio_scale_shift_table[None] + audio_temb[:, :2]).chunk(2, dim=1)
        audio_hidden = audio_hidden * (1 + audio_scale.squeeze(1)) + audio_shift.squeeze(1)
        audio_output = self.audio_proj_out(audio_hidden)

        return video_output, audio_output


def load_ltx2_weights(model: NeuronLTX2Transformer, checkpoint_dir: str) -> None:
    """Load weights from sharded safetensors."""
    import glob
    from safetensors.torch import load_file

    state_dict = {}
    for f in sorted(glob.glob(f"{checkpoint_dir}/diffusion_pytorch_model-*.safetensors")):
        logger.info(f"Loading {f}")
        part = load_file(f)
        state_dict.update(part)

    # Map weights
    model_state = model.state_dict()
    mapped = {}

    for key in model_state.keys():
        if key in state_dict:
            mapped[key] = state_dict[key]
        else:
            # Try mapping with prefixes
            src_key = key
            if src_key in state_dict:
                mapped[key] = state_dict[src_key]

    model.load_state_dict(mapped, strict=False)
    logger.info(f"Loaded {len(mapped)}/{len(model_state)} weights")
