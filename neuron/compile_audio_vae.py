#!/usr/bin/env python3
"""
LTX-2 Audio VAE Compilation for Neuron

2D VAE for audio mel-spectrogram encoding/decoding.
~102MB model, uses Conv2d with causal padding.
"""

import os
import sys
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LTX2-AudioVAE")

MODEL_DIR = "/home/ubuntu/models/ltx2/audio_vae"
OUTPUT_DIR = "/home/ubuntu/ltx2/neuron/compiled"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mel_bins", type=int, default=64, help="Mel frequency bins")
    parser.add_argument("--time_frames", type=int, default=256, help="Time frames")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--encoder_only", action="store_true", help="Compile encoder only")
    parser.add_argument("--decoder_only", action="store_true", help="Compile decoder only")
    return parser.parse_args()


class CausalConv2d(nn.Module):
    """Causal Conv2d with padding on height (time) axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] where H is time (causal), W is frequency
        # Causal padding: pad only past on H axis
        # Standard padding on W axis
        pad_h = self.kernel_size - 1  # Causal: all padding on top
        pad_w = self.padding

        # F.pad format: (left, right, top, bottom)
        x = F.pad(x, (pad_w, pad_w, pad_h, 0))
        return self.conv(x)


class ResBlock2d(nn.Module):
    """Residual block with optional channel change."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = CausalConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Pixel norm (group norm with 1 group)
        x = F.group_norm(x, 1)
        x = F.silu(x)
        x = self.conv1(x)

        x = F.group_norm(x, 1)
        x = F.silu(x)
        x = self.conv2(x)

        if self.nin_shortcut is not None:
            residual = self.nin_shortcut(residual)

        return x + residual


class Downsample2d(nn.Module):
    """Downsample with strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2d(nn.Module):
    """Upsample with interpolation + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class AudioEncoder(nn.Module):
    """Audio VAE Encoder."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 128,
        ch_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 8,
        double_z: bool = True,
    ):
        super().__init__()
        self.conv_in = CausalConv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling blocks
        self.down = nn.ModuleList()
        in_ch = base_channels
        for i, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            block = nn.ModuleDict()

            # Residual blocks
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                blocks.append(ResBlock2d(in_ch if j == 0 else out_ch, out_ch))
            block['block'] = blocks

            # Downsample (except last)
            if i < len(ch_mult) - 1:
                block['downsample'] = Downsample2d(out_ch)

            self.down.append(block)
            in_ch = out_ch

        # Mid blocks
        self.mid = nn.ModuleDict({
            'block_1': ResBlock2d(in_ch, in_ch),
            'block_2': ResBlock2d(in_ch, in_ch),
        })

        # Output
        out_channels = latent_channels * 2 if double_z else latent_channels
        self.conv_out = CausalConv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, H, W] mel spectrogram
        x = self.conv_in(x)

        # Downsample
        for down_block in self.down:
            for block in down_block['block']:
                x = block(x)
            if 'downsample' in down_block:
                x = down_block['downsample'](x)

        # Mid
        x = self.mid['block_1'](x)
        x = self.mid['block_2'](x)

        # Output
        x = F.group_norm(x, 1)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


class AudioDecoder(nn.Module):
    """Audio VAE Decoder.

    Weight structure (from safetensors):
    - up.0: 256->128 channels (final output stage)
    - up.1: 512->256 channels (middle stage, has upsample)
    - up.2: 512->512 channels (first stage after mid, has upsample)

    Processing order: mid -> up.2 -> up.1 -> up.0 -> conv_out
    """

    def __init__(
        self,
        out_channels: int = 2,
        base_channels: int = 128,
        ch_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 8,
    ):
        super().__init__()

        # Start from highest channel count
        in_ch = base_channels * ch_mult[-1]  # 512
        self.conv_in = CausalConv2d(latent_channels, in_ch, kernel_size=3, padding=1)

        # Mid blocks
        self.mid = nn.ModuleDict({
            'block_1': ResBlock2d(in_ch, in_ch),
            'block_2': ResBlock2d(in_ch, in_ch),
        })

        # Build up blocks indexed to match weights
        # up.0: 256->128, up.1: 512->256 + upsample, up.2: 512->512 + upsample
        self.up = nn.ModuleList()

        # ch_mult = (1, 2, 4) means channels: 128, 256, 512
        # We build in weight index order: 0, 1, 2
        ch_configs = [
            (256, 128, False),   # up.0: in=256, out=128, no upsample
            (512, 256, True),    # up.1: in=512, out=256, has upsample
            (512, 512, True),    # up.2: in=512, out=512, has upsample
        ]

        for in_channels, out_channels_stage, has_upsample in ch_configs:
            block = nn.ModuleDict()

            # 3 res blocks per stage
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                block_in = in_channels if j == 0 else out_channels_stage
                blocks.append(ResBlock2d(block_in, out_channels_stage))
            block['block'] = blocks

            if has_upsample:
                block['upsample'] = Upsample2d(out_channels_stage)

            self.up.append(block)

        # Output
        self.conv_out = CausalConv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, 8, H, W] latent
        x = self.conv_in(z)

        # Mid
        x = self.mid['block_1'](x)
        x = self.mid['block_2'](x)

        # Upsample - process from up.2 -> up.1 -> up.0
        for idx in [2, 1, 0]:
            up_block = self.up[idx]
            for block in up_block['block']:
                x = block(x)
            if 'upsample' in up_block:
                x = up_block['upsample'](x)

        # Output
        x = F.group_norm(x, 1)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


class AudioVAE(nn.Module):
    """Full Audio VAE (Encoder + Decoder)."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 128,
        ch_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 8,
    ):
        super().__init__()
        self.encoder = AudioEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
            double_z=True,
        )
        self.decoder = AudioDecoder(
            out_channels=out_channels,
            base_channels=base_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )

        # Latent statistics
        self.register_buffer('latents_mean', torch.zeros(128))
        self.register_buffer('latents_std', torch.ones(128))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        # Split into mean and logvar
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encode(x)
        # For inference, just use mean (no sampling)
        return self.decode(mean)


def load_weights(model: nn.Module, model_dir: str, prefix: str = ""):
    """Load weights from safetensors."""
    from safetensors.torch import load_file

    weights_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    logger.info(f"Loading weights from {weights_path}")

    state_dict = load_file(weights_path)
    model_state = model.state_dict()

    loaded = 0
    for key in model_state.keys():
        # Map model key to weight key
        weight_key = prefix + key if prefix else key

        if weight_key in state_dict:
            if model_state[key].shape == state_dict[weight_key].shape:
                model_state[key] = state_dict[weight_key]
                loaded += 1
            else:
                logger.warning(f"Shape mismatch: {key} model={model_state[key].shape} weight={state_dict[weight_key].shape}")
        else:
            logger.debug(f"Missing key: {weight_key}")

    model.load_state_dict(model_state)
    logger.info(f"Loaded {loaded}/{len(model_state)} weights")
    return loaded


def test_cpu_forward(args):
    """Test forward pass on CPU."""
    logger.info("Testing CPU forward...")

    model = AudioVAE()
    model = model.to(torch.bfloat16)
    model.eval()

    # Input: [B, 2, mel_bins, time_frames]
    x = torch.randn(args.batch_size, 2, args.mel_bins, args.time_frames, dtype=torch.bfloat16)
    logger.info(f"Input shape: {x.shape}")

    with torch.no_grad():
        # Test encoder
        mean, logvar = model.encode(x)
        logger.info(f"Encoded mean shape: {mean.shape}")

        # Test decoder
        recon = model.decode(mean)
        logger.info(f"Decoded shape: {recon.shape}")

    logger.info("CPU forward OK!")
    return True


def compile_encoder(args):
    """Compile encoder only."""
    import torch_neuronx

    logger.info("Compiling encoder...")
    model = AudioEncoder()
    model = model.to(torch.bfloat16)

    if args.load_weights:
        load_weights(model, args.model_dir, prefix="encoder.")

    model.eval()

    x = torch.randn(args.batch_size, 2, args.mel_bins, args.time_frames, dtype=torch.bfloat16)

    try:
        traced = torch_neuronx.trace(
            model, x,
            compiler_args=["--model-type=unet-inference", "-O2"],
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"audio_vae_encoder_bs{args.batch_size}_mel{args.mel_bins}_t{args.time_frames}.pt"
        )
        torch.jit.save(traced, output_path)
        logger.info(f"Saved encoder to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Encoder compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compile_decoder(args):
    """Compile decoder only."""
    import torch_neuronx

    logger.info("Compiling decoder...")
    model = AudioDecoder()
    model = model.to(torch.bfloat16)

    if args.load_weights:
        load_weights(model, args.model_dir, prefix="decoder.")

    model.eval()

    # Latent shape: encoder output is 1/4 spatial size (2 downsamples)
    latent_h = args.mel_bins // 4
    latent_w = args.time_frames // 4
    z = torch.randn(args.batch_size, 8, latent_h, latent_w, dtype=torch.bfloat16)

    logger.info(f"Latent input shape: {z.shape}")

    try:
        traced = torch_neuronx.trace(
            model, z,
            compiler_args=["--model-type=unet-inference", "-O2"],
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"audio_vae_decoder_bs{args.batch_size}_mel{args.mel_bins}_t{args.time_frames}.pt"
        )
        torch.jit.save(traced, output_path)
        logger.info(f"Saved decoder to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Decoder compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compile_full(args):
    """Compile full VAE."""
    import torch_neuronx

    logger.info("Compiling full VAE...")
    model = AudioVAE()
    model = model.to(torch.bfloat16)

    if args.load_weights:
        load_weights(model, args.model_dir)

    model.eval()

    x = torch.randn(args.batch_size, 2, args.mel_bins, args.time_frames, dtype=torch.bfloat16)

    try:
        traced = torch_neuronx.trace(
            model, x,
            compiler_args=["--model-type=unet-inference", "-O2"],
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"audio_vae_full_bs{args.batch_size}_mel{args.mel_bins}_t{args.time_frames}.pt"
        )
        torch.jit.save(traced, output_path)
        logger.info(f"Saved full VAE to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Full VAE compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("LTX-2 Audio VAE Compilation")
    logger.info("=" * 60)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Mel bins: {args.mel_bins}")
    logger.info(f"Time frames: {args.time_frames}")

    # Test CPU first
    if not test_cpu_forward(args):
        return 1

    os.environ["NEURON_CC_FLAGS"] = "--model-type=unet-inference -O2"

    success = True
    if args.encoder_only:
        success = compile_encoder(args)
    elif args.decoder_only:
        success = compile_decoder(args)
    else:
        # Compile both separately for flexibility
        success = compile_encoder(args) and compile_decoder(args)

    if success:
        logger.info("Compilation succeeded!")
        return 0
    else:
        logger.error("Compilation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
