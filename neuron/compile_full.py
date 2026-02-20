#!/usr/bin/env python3
"""
LTX-2 Full Transformer Compilation with Weight Loading

Usage:
    python compile_full.py --tp_degree 4 --num_layers 48 --load_weights
"""

import os
import sys
import json
import argparse
import logging
import math
import glob
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LTX2-Full")

MODEL_DIR = "/home/ubuntu/models/ltx2/transformer"
OUTPUT_DIR = "/home/ubuntu/ltx2/neuron/compiled"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_degree", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    return parser.parse_args()


class RMSNorm(nn.Module):
    """RMS Normalization for QK norm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x_norm).to(dtype)


class LTX2Attention(nn.Module):
    """LTX-2 Attention module."""

    def __init__(
        self,
        query_dim: int = 4096,
        cross_attention_dim: Optional[int] = None,
        num_heads: int = 32,
        head_dim: int = 128,
        bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.cross_attention_dim = cross_attention_dim or query_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim, bias=bias),
        )

        # QK normalization (RMS norm across heads)
        self.norm_q = RMSNorm(self.inner_dim)
        self.norm_k = RMSNorm(self.inner_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # QK norm
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention (compute in float32 for stability, convert back)
        dtype = q.dtype
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.float())

        # Reshape back and convert to original dtype
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1).to(dtype)
        out = self.to_out(out)

        return out


class LTX2FeedForward(nn.Module):
    """GEGLU FeedForward."""

    def __init__(self, dim: int = 4096, inner_dim: int = 16384):
        super().__init__()
        # GEGLU: proj to 2x, split, multiply
        self.net = nn.ModuleList([
            nn.Linear(dim, inner_dim * 2, bias=True),  # net.0.proj
            nn.Identity(),  # placeholder for GELU
            nn.Linear(inner_dim, dim, bias=True),  # net.2
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.net[0](x)
        hidden, gate = hidden.chunk(2, dim=-1)
        hidden = hidden * F.gelu(gate, approximate="tanh")
        return self.net[2](hidden)


class LTX2TransformerBlock(nn.Module):
    """Single LTX-2 Transformer Block (video-only for now)."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        cross_attention_dim: int = 4096,
        ff_inner_dim: int = 16384,
    ):
        super().__init__()

        # Self attention
        self.attn1 = LTX2Attention(
            query_dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Cross attention with text
        self.attn2 = LTX2Attention(
            query_dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # FFN
        self.ff = LTX2FeedForward(dim=hidden_size, inner_dim=ff_inner_dim // 2)

        # AdaLN scale/shift table
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + temb.unsqueeze(1)
        ).chunk(6, dim=1)

        # Self attention
        norm_hidden = hidden_states * (1 + scale_msa) + shift_msa
        attn_out = self.attn1(norm_hidden)
        hidden_states = hidden_states + gate_msa * attn_out

        # Cross attention
        attn_out = self.attn2(hidden_states, encoder_hidden_states)
        hidden_states = hidden_states + attn_out

        # FFN
        norm_hidden = hidden_states * (1 + scale_mlp) + shift_mlp
        ff_out = self.ff(norm_hidden)
        hidden_states = hidden_states + gate_mlp * ff_out

        return hidden_states


class LTX2VideoTransformer(nn.Module):
    """LTX-2 Video Transformer (video branch only)."""

    def __init__(
        self,
        num_layers: int = 48,
        hidden_size: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        cross_attention_dim: int = 4096,
        caption_channels: int = 3840,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Caption projection
        self.caption_projection = nn.Sequential(
            nn.Linear(caption_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Input/output projections
        self.proj_in = nn.Linear(in_channels, hidden_size)
        self.proj_out = nn.Linear(hidden_size, out_channels)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            LTX2TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(num_layers)
        ])

        # Final scale/shift
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))

    def get_timestep_embedding(self, timesteps: torch.Tensor, dim: int = 256) -> torch.Tensor:
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb.to(torch.bfloat16)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, N, C] flattened video latents
            timestep: [B] diffusion timestep
            encoder_hidden_states: [B, S, D] text embeddings

        Returns:
            [B, N, C] noise prediction
        """
        # Timestep embedding
        t_emb = self.get_timestep_embedding(timestep)
        temb = self.time_embed(t_emb)

        # Project text
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)

        # Project input
        hidden = self.proj_in(latents)

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden = block(hidden, encoder_hidden_states, temb)

        # Final norm and projection
        shift, scale = (self.scale_shift_table[None] + temb.unsqueeze(1)[:, :2]).chunk(2, dim=1)
        hidden = hidden * (1 + scale) + shift
        output = self.proj_out(hidden)

        return output


def load_weights(model: LTX2VideoTransformer, model_dir: str) -> Dict[str, int]:
    """Load weights from sharded safetensors."""
    from safetensors.torch import load_file

    logger.info(f"Loading weights from {model_dir}")

    # Load index
    with open(os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Group by shard file
    shard_to_keys = {}
    for key, shard in weight_map.items():
        if shard not in shard_to_keys:
            shard_to_keys[shard] = []
        shard_to_keys[shard].append(key)

    # Load and map weights
    model_state = model.state_dict()
    loaded_count = 0
    missing_keys = []

    for shard_file, keys in shard_to_keys.items():
        shard_path = os.path.join(model_dir, shard_file)
        logger.info(f"Loading {shard_file}...")
        shard_weights = load_file(shard_path)

        for orig_key in keys:
            # Map original key to model key
            model_key = map_weight_key(orig_key)

            if model_key and model_key in model_state:
                if model_state[model_key].shape == shard_weights[orig_key].shape:
                    model_state[model_key] = shard_weights[orig_key]
                    loaded_count += 1
                else:
                    logger.warning(f"Shape mismatch: {model_key} "
                                 f"model={model_state[model_key].shape} "
                                 f"weight={shard_weights[orig_key].shape}")

    model.load_state_dict(model_state, strict=False)
    logger.info(f"Loaded {loaded_count}/{len(model_state)} weights")

    return {"loaded": loaded_count, "total": len(model_state)}


def map_weight_key(orig_key: str) -> Optional[str]:
    """Map original weight key to model key."""
    # Direct mappings
    if orig_key.startswith("transformer_blocks."):
        # transformer_blocks.X.attn1.to_q.weight -> transformer_blocks.X.attn1.to_q.weight
        return orig_key

    if orig_key == "proj_in.weight" or orig_key == "proj_in.bias":
        return orig_key

    if orig_key == "proj_out.weight" or orig_key == "proj_out.bias":
        return orig_key

    if orig_key == "scale_shift_table":
        return orig_key

    if orig_key.startswith("time_embed."):
        # time_embed.emb.timestep_embedder.linear_1.weight -> time_embed.0.weight
        if "linear_1" in orig_key:
            return orig_key.replace("emb.timestep_embedder.linear_1", "0")
        if "linear_2" in orig_key:
            return orig_key.replace("emb.timestep_embedder.linear_2", "2")
        if orig_key.startswith("time_embed.linear"):
            # time_embed.linear.weight -> Not directly mapped (extra projection)
            return None

    if orig_key.startswith("caption_projection."):
        # caption_projection.linear_1.weight -> caption_projection.0.weight
        if "linear_1" in orig_key:
            return orig_key.replace("linear_1", "0")
        if "linear_2" in orig_key:
            return orig_key.replace("linear_2", "2")

    # Skip audio-related weights for now
    if "audio" in orig_key.lower():
        return None

    return None


# Global config for model factory
_CONFIG = {
    "num_layers": 48,
    "seq_len": 512,
    "load_weights": False,
    "model_dir": MODEL_DIR,
}


def create_model():
    """Model factory for parallel_model_trace."""
    model = LTX2VideoTransformer(num_layers=_CONFIG["num_layers"])
    model = model.to(torch.bfloat16)

    if _CONFIG["load_weights"]:
        load_weights(model, _CONFIG["model_dir"])

    model.eval()
    return model, None


def main():
    global _CONFIG
    args = parse_args()

    logger.info("=" * 60)
    logger.info("LTX-2 Full Transformer Compilation")
    logger.info("=" * 60)
    logger.info(f"TP Degree: {args.tp_degree}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Seq Length: {args.seq_len}")
    logger.info(f"Num Layers: {args.num_layers}")
    logger.info(f"Load Weights: {args.load_weights}")

    # Update config
    _CONFIG["num_layers"] = args.num_layers
    _CONFIG["seq_len"] = args.seq_len
    _CONFIG["load_weights"] = args.load_weights
    _CONFIG["model_dir"] = args.model_dir

    # Test CPU forward first
    logger.info("Testing CPU forward...")
    model = LTX2VideoTransformer(num_layers=min(args.num_layers, 2))
    model = model.to(torch.bfloat16)
    model.eval()

    latents = torch.randn(args.batch_size, 64, 128, dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (args.batch_size,), dtype=torch.long)
    encoder_states = torch.randn(args.batch_size, 32, 3840, dtype=torch.bfloat16)

    with torch.no_grad():
        output = model(latents, timestep, encoder_states)
    logger.info(f"CPU forward OK: output shape = {output.shape}")

    # Set environment
    os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer -O2"

    # Import NxD
    from neuronx_distributed.trace import parallel_model_trace, parallel_model_save

    # Sample inputs for tracing
    latents = torch.randn(args.batch_size, args.seq_len, 128, dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (args.batch_size,), dtype=torch.long)
    encoder_states = torch.randn(args.batch_size, 256, 3840, dtype=torch.bfloat16)
    sample_inputs = (latents, timestep, encoder_states)

    logger.info(f"Sample inputs: latents={latents.shape}, encoder={encoder_states.shape}")
    logger.info("Starting parallel_model_trace...")

    try:
        traced = parallel_model_trace(
            create_model,
            sample_inputs,
            tp_degree=args.tp_degree,
            compiler_args=["--model-type=transformer", "-O2"],
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"ltx2_video_tp{args.tp_degree}_L{args.num_layers}"
        )

        parallel_model_save(traced, output_path)
        logger.info(f"Saved to {output_path}")
        logger.info("Compilation succeeded!")

    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
