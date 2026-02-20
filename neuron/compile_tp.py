#!/usr/bin/env python3
"""
LTX-2 Transformer TP Compilation Script

This script must be run separately (not after single-core compile)
to avoid XLA runtime conflicts.

Usage:
    python compile_tp.py --tp_degree 4 --batch_size 1 --seq_len 1024
"""

import os
import sys
import json
import argparse
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LTX2-TP-Compile")

MODEL_DIR = "/home/ubuntu/models/ltx2/transformer"
OUTPUT_DIR = "/home/ubuntu/ltx2/neuron/compiled"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_degree", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers for test compilation")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


class SimplifiedLTX2Block(nn.Module):
    """Simplified transformer block."""

    def __init__(self, hidden_size=4096, num_heads=32, head_dim=128, ff_dim=16384):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_dim = ff_dim

        # Attention
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.to_out = nn.Linear(hidden_size, hidden_size)

        # QK norm
        self.norm_q = nn.LayerNorm(head_dim, elementwise_affine=False)
        self.norm_k = nn.LayerNorm(head_dim, elementwise_affine=False)

        # FFN (GEGLU)
        self.ff_proj = nn.Linear(hidden_size, ff_dim * 2)
        self.ff_out = nn.Linear(ff_dim, hidden_size)

        # AdaLN
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # AdaLN
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + temb.unsqueeze(1)
        ).chunk(6, dim=1)

        # Self-attention
        norm_hidden = hidden_states * (1 + scale_msa) + shift_msa

        q = self.to_q(norm_hidden)
        k = self.to_k(norm_hidden)
        v = self.to_v(norm_hidden)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_out = self.to_out(attn_out)

        hidden_states = hidden_states + gate_msa * attn_out

        # FFN
        norm_hidden = hidden_states * (1 + scale_mlp) + shift_mlp
        ff = self.ff_proj(norm_hidden)
        ff, gate = ff.chunk(2, dim=-1)
        ff = ff * F.gelu(gate, approximate="tanh")
        ff = self.ff_out(ff)

        hidden_states = hidden_states + gate_mlp * ff

        return hidden_states


class SimplifiedLTX2Transformer(nn.Module):
    """Simplified LTX-2 for TP compilation."""

    def __init__(self, num_layers: int = 2, hidden_size: int = 4096, in_channels: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Projections
        self.proj_in = nn.Linear(in_channels, hidden_size)
        self.proj_out = nn.Linear(hidden_size, in_channels)

        # Blocks
        self.blocks = nn.ModuleList([
            SimplifiedLTX2Block(hidden_size=hidden_size)
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

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t_emb = self.get_timestep_embedding(timestep)
        temb = self.time_embed(t_emb)

        hidden = self.proj_in(latents)

        for block in self.blocks:
            hidden = block(hidden, temb)

        shift, scale = (self.scale_shift_table[None] + temb.unsqueeze(1)[:, :2]).chunk(2, dim=1)
        hidden = hidden * (1 + scale) + shift
        output = self.proj_out(hidden)

        return output


# Global config for model factory (needs to be picklable)
_COMPILE_CONFIG = {
    "num_layers": 2,
}


def create_model():
    """Model factory function at module level for pickle support.

    Returns:
        (model, input_output_alias) tuple as expected by parallel_model_trace
    """
    model = SimplifiedLTX2Transformer(num_layers=_COMPILE_CONFIG["num_layers"])
    model = model.to(torch.bfloat16)
    model.eval()
    # input_output_alias is used for in-place operations (like KV cache)
    # For this model, we don't have any
    return model, None


def main():
    global _COMPILE_CONFIG
    args = parse_args()

    logger.info("=" * 60)
    logger.info("LTX-2 Transformer TP Compilation")
    logger.info("=" * 60)
    logger.info(f"TP Degree: {args.tp_degree}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Seq Length: {args.seq_len}")
    logger.info(f"Num Layers: {args.num_layers}")

    # Set global config
    _COMPILE_CONFIG["num_layers"] = args.num_layers

    # Set environment
    os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer -O2"

    # Import NxD (do NOT use any XLA before this)
    from neuronx_distributed.trace import parallel_model_trace, parallel_model_save

    # Sample inputs
    latents = torch.randn(args.batch_size, args.seq_len, 128, dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (args.batch_size,), dtype=torch.long)
    sample_inputs = (latents, timestep)

    logger.info(f"Sample input shapes: latents={latents.shape}, timestep={timestep.shape}")
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
            f"ltx2_tp{args.tp_degree}_bs{args.batch_size}_seq{args.seq_len}_L{args.num_layers}"
        )

        # Use parallel_model_save for TP models
        parallel_model_save(traced, output_path)

        logger.info(f"Saved to {output_path}")
        logger.info("TP compilation succeeded!")

    except Exception as e:
        logger.error(f"TP compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
