# LTX-2 Neuron Configuration
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class LTX2NeuronConfig:
    """Configuration for LTX-2 Neuron compilation."""

    # Model architecture
    num_layers: int = 48
    hidden_size: int = 4096  # video hidden size
    audio_hidden_size: int = 2048
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    cross_attention_dim: int = 4096
    audio_cross_attention_dim: int = 2048
    caption_channels: int = 3840
    in_channels: int = 128
    out_channels: int = 128
    audio_in_channels: int = 128
    audio_out_channels: int = 128

    # FFN
    ff_inner_dim: int = 16384  # 4 * hidden_size
    audio_ff_inner_dim: int = 8192  # 4 * audio_hidden_size
    activation_fn: str = "gelu-approximate"

    # Normalization
    norm_eps: float = 1e-6
    qk_norm: str = "rms_norm_across_heads"

    # Position embedding
    rope_theta: float = 10000.0
    rope_type: str = "split"
    rope_double_precision: bool = True
    pos_embed_max_pos: int = 20
    base_height: int = 2048
    base_width: int = 2048

    # VAE scale factors
    vae_scale_factors: Tuple[int, int, int] = (8, 32, 32)

    # Neuron specific
    tp_degree: int = 4
    torch_dtype: str = "bfloat16"

    # Compilation
    batch_size: int = 1
    max_num_frames: int = 121  # (8*15 + 1)
    max_height: int = 544
    max_width: int = 960

    @property
    def max_video_tokens(self) -> int:
        """Maximum number of video latent tokens."""
        t = (self.max_num_frames - 1) // self.vae_scale_factors[0] + 1
        h = self.max_height // self.vae_scale_factors[1]
        w = self.max_width // self.vae_scale_factors[2]
        return t * h * w

    @property
    def max_audio_tokens(self) -> int:
        """Maximum number of audio tokens (approximate)."""
        # Audio is ~5.12 seconds at 16kHz with hop_length=160
        return 512


DEFAULT_CONFIG = LTX2NeuronConfig()
