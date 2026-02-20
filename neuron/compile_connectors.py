#!/usr/bin/env python3
"""
LTX-2 Text Connectors Compilation for Neuron

Connects text encoder output to video/audio transformer branches.
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
logger = logging.getLogger("LTX2-Connectors")

MODEL_DIR = "/home/ubuntu/models/ltx2/connectors"
OUTPUT_DIR = "/home/ubuntu/ltx2/neuron/compiled"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--text_seq_len", type=int, default=256, help="Text sequence length")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--load_weights", action="store_true")
    return parser.parse_args()


class RMSNorm(nn.Module):
    """RMS Normalization."""
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


class ConnectorAttention(nn.Module):
    """Self-attention for connector."""

    def __init__(
        self,
        dim: int = 3840,
        num_heads: int = 30,
        head_dim: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim, bias=True),
        )

        self.norm_q = RMSNorm(self.inner_dim)
        self.norm_k = RMSNorm(self.inner_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # QK norm
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        dtype = q.dtype
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.float()).to(dtype)

        # Reshape back
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.to_out(out)

        return out


class FFNProj(nn.Module):
    """FFN projection layer with GELU activation."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class ConnectorFeedForward(nn.Module):
    """FFN matching original weight structure: net.0.proj + GELU + net.2"""

    def __init__(self, dim: int = 3840, inner_dim: int = 15360):
        super().__init__()
        # Original structure:
        # net.0.proj: [15360, 3840] + GELU activation
        # net.2: [3840, 15360]
        self.net = nn.ModuleList([
            FFNProj(dim, inner_dim),  # net.0.proj with GELU
            nn.Identity(),
            nn.Linear(inner_dim, dim, bias=True),  # net.2
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.net[0](x)  # proj + GELU
        return self.net[2](hidden)


class ConnectorTransformerBlock(nn.Module):
    """Single connector transformer block."""

    def __init__(self, dim: int = 3840, num_heads: int = 30, head_dim: int = 128):
        super().__init__()
        self.attn1 = ConnectorAttention(dim, num_heads, head_dim)
        self.ff = ConnectorFeedForward(dim, dim * 4)  # 3840 * 4 = 15360

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(x)
        x = x + self.ff(x)
        return x


class SingleConnector(nn.Module):
    """Single connector (video or audio)."""

    def __init__(
        self,
        dim: int = 3840,
        num_heads: int = 30,
        head_dim: int = 128,
        num_layers: int = 2,
        num_registers: int = 128,
    ):
        super().__init__()
        self.learnable_registers = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.transformer_blocks = nn.ModuleList([
            ConnectorTransformerBlock(dim, num_heads, head_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] text embeddings

        Returns:
            [B, num_registers, D] compressed representations
        """
        batch_size = x.shape[0]

        # Add learnable registers
        registers = self.learnable_registers.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([registers, x], dim=1)  # [B, num_registers + S, D]

        # Transform
        for block in self.transformer_blocks:
            x = block(x)

        # Return only register outputs
        return x[:, :self.learnable_registers.shape[0], :]


class LTX2TextConnectors(nn.Module):
    """LTX-2 Text Connectors for video and audio."""

    def __init__(
        self,
        caption_channels: int = 3840,
        text_proj_in_factor: int = 49,
        num_heads: int = 30,
        head_dim: int = 128,
        num_layers: int = 2,
        num_registers: int = 128,
    ):
        super().__init__()

        # Text input projection (from Gemma hidden states)
        # 188160 = 3840 * 49 (concatenated hidden states from multiple layers)
        self.text_proj_in = nn.Linear(caption_channels * text_proj_in_factor, caption_channels, bias=False)

        # Video connector
        self.video_connector = SingleConnector(
            dim=caption_channels,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            num_registers=num_registers,
        )

        # Audio connector
        self.audio_connector = SingleConnector(
            dim=caption_channels,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            num_registers=num_registers,
        )

    def forward(self, text_hidden_states: torch.Tensor):
        """
        Args:
            text_hidden_states: [B, S, D*49] concatenated text encoder hidden states

        Returns:
            video_embeds: [B, 128, 3840]
            audio_embeds: [B, 128, 3840]
        """
        # Project text
        text_embeds = self.text_proj_in(text_hidden_states)

        # Get video and audio embeddings
        video_embeds = self.video_connector(text_embeds)
        audio_embeds = self.audio_connector(text_embeds)

        return video_embeds, audio_embeds


def load_weights(model: LTX2TextConnectors, model_dir: str):
    """Load weights from safetensors."""
    from safetensors.torch import load_file

    weights_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    logger.info(f"Loading weights from {weights_path}")

    state_dict = load_file(weights_path)
    model_state = model.state_dict()

    loaded = 0
    for key in model_state.keys():
        if key in state_dict:
            if model_state[key].shape == state_dict[key].shape:
                model_state[key] = state_dict[key]
                loaded += 1
            else:
                logger.warning(f"Shape mismatch: {key} model={model_state[key].shape} weight={state_dict[key].shape}")
        else:
            logger.warning(f"Missing key: {key}")

    model.load_state_dict(model_state)
    logger.info(f"Loaded {loaded}/{len(model_state)} weights")


def test_cpu_forward(args):
    """Test forward pass on CPU."""
    logger.info("Testing CPU forward...")

    model = LTX2TextConnectors()
    model = model.to(torch.bfloat16)
    model.eval()

    # Input: concatenated hidden states from text encoder
    # Shape: [B, seq_len, 3840 * 49]
    x = torch.randn(args.batch_size, args.text_seq_len, 3840 * 49, dtype=torch.bfloat16)
    logger.info(f"Input shape: {x.shape}")

    with torch.no_grad():
        video_embeds, audio_embeds = model(x)

    logger.info(f"Video embeds shape: {video_embeds.shape}")
    logger.info(f"Audio embeds shape: {audio_embeds.shape}")
    logger.info("CPU forward OK!")

    return True


def compile_model(args):
    """Compile model with torch_neuronx."""
    import torch_neuronx

    logger.info("Creating model...")
    model = LTX2TextConnectors()
    model = model.to(torch.bfloat16)

    if args.load_weights:
        load_weights(model, args.model_dir)

    model.eval()

    # Sample input
    x = torch.randn(args.batch_size, args.text_seq_len, 3840 * 49, dtype=torch.bfloat16)

    logger.info(f"Tracing with input shape: {x.shape}")
    logger.info("Starting compilation...")

    try:
        traced = torch_neuronx.trace(
            model,
            x,
            compiler_args=[
                "--model-type=transformer",
                "-O2",
            ],
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"connectors_bs{args.batch_size}_seq{args.text_seq_len}.pt"
        )
        torch.jit.save(traced, output_path)
        logger.info(f"Saved to {output_path}")

        return True

    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("LTX-2 Text Connectors Compilation")
    logger.info("=" * 60)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Text seq len: {args.text_seq_len}")

    # Test CPU first
    if not test_cpu_forward(args):
        return 1

    # Set Neuron flags
    os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer -O2"

    # Compile
    if compile_model(args):
        logger.info("Compilation succeeded!")
        return 0
    else:
        logger.error("Compilation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
