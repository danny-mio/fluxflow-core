"""
Encoder models for FluxFlow.

Contains:
- BertTextEncoder: Text encoder using DistilBERT
- ImageEncoder: Image encoder with progressive downsampling
"""

import os

import safetensors.torch
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from transformers import DistilBertConfig, DistilBertModel

from .activations import BezierActivation, xavier_init
from .conditioning import SPADE, LeanContext2D


class BertTextEncoder(nn.Module):
    """
    Text encoder using DistilBERT transformer.

    Encodes text input_ids to dense embeddings suitable for conditioning.

    Args:
        embed_dim: Output embedding dimension
        pretrain_model: Optional pretrained model name (e.g., 'distilbert-base-uncased')
                       If None, initializes from scratch
    """

    def __init__(self, embed_dim, pretrain_model=None):
        super(BertTextEncoder, self).__init__()
        self.apply_bezier_activation = BezierActivation()

        if pretrain_model is None:
            self.language_model = DistilBertModel(
                DistilBertConfig(
                    vocab_size=30522,
                    dim=768,
                    n_layers=6,
                    n_heads=12,
                    hidden_dim=3072,
                    max_position_embeddings=512,
                    dropout=0.1,
                    attention_dropout=0.1,
                    activation="gelu",
                )
            )
            self.language_model.init_weights()
        else:
            # Pre-trained language model for natural language understanding
            self.load_language_model(pretrain_model)

        self.ouput_layer = nn.Sequential(
            nn.Linear(self.language_model.config.hidden_size, 2560),  # 512 * 5 for Bezier
            nn.LayerNorm(2560),
            self.apply_bezier_activation,
            nn.Linear(512, embed_dim * 5),
            nn.LayerNorm(embed_dim * 5),
            self.apply_bezier_activation,
        )

        # Initialize weights using Xavier initialization
        self.apply(xavier_init)

    def load_language_model(self, pretrain_model):
        """Load pretrained DistilBERT model and freeze weights."""
        self.language_model = DistilBertModel.from_pretrained(
            pretrain_model, cache_dir="./_cache", local_files_only=False
        )

        # Freeze language model weights to avoid updating during training
        for param in self.language_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]

        Returns:
            Text embeddings [B, embed_dim]
        """
        # Extract language model outputs
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)

        # Use the last hidden state
        last_hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # Aggregate embeddings by taking mean across sequence
        sentence_embedding = last_hidden_state.mean(dim=1)  # [B, hidden_size]

        # Apply output layer with Bezier activation
        out = self.ouput_layer(sentence_embedding)  # [B, embed_dim]

        return out

    def save_checkpoint(self, path, save_language_model=False):
        """
        Save encoder checkpoint to safetensors file.

        Args:
            path: Path to save checkpoint (file or directory)
            save_language_model: If True, save full model including DistilBERT weights.
                               If False (default), only save output layer weights.
        """
        # Determine file path
        if os.path.isdir(path):
            file_path = os.path.join(path, "text_encoder.safetensors")
        else:
            file_path = path
            # Create parent directory if needed
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        # Build state dict
        if save_language_model:
            state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        else:
            # Only save output layer (language model can be reloaded from pretrained)
            state_dict = {
                k: v.cpu()
                for k, v in self.state_dict().items()
                if not k.startswith("language_model.")
            }

        safetensors.torch.save_file(state_dict, file_path)
        return file_path

    @classmethod
    def from_checkpoint(
        cls, path, embed_dim, pretrain_model="distilbert-base-uncased", device=None
    ):
        """
        Load encoder from checkpoint file.

        Args:
            path: Path to checkpoint file or directory containing text_encoder.safetensors
            embed_dim: Output embedding dimension (must match saved model)
            pretrain_model: Pretrained model name for DistilBERT backbone.
                          Use None to load language_model from checkpoint.
            device: Device to load model to (default: cpu)

        Returns:
            BertTextEncoder instance with loaded weights

        Example:
            >>> encoder = BertTextEncoder.from_checkpoint(
            ...     'checkpoints/text_encoder.safetensors',
            ...     embed_dim=256
            ... )
        """
        # Determine file path
        if os.path.isdir(path):
            file_path = os.path.join(path, "text_encoder.safetensors")
        else:
            file_path = path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint not found: {file_path}")

        # Create model instance
        encoder = cls(embed_dim=embed_dim, pretrain_model=pretrain_model)

        # Load state dict
        state_dict = safetensors.torch.load_file(file_path)

        # Load weights (strict=False allows missing language_model keys)
        encoder.load_state_dict(state_dict, strict=False)

        # Move to device if specified
        if device is not None:
            encoder = encoder.to(device)

        return encoder


class ImageEncoder(nn.Module):
    """
    Image encoder with progressive downsampling and Bezier activations.

    Encodes images to embeddings matching text_embedding_dim for contrastive learning.

    Args:
        img_channels: Input image channels (default: 3)
        text_embedding_dim: Output embedding dimension
        feature_maps: Base feature map channels (default: 128)
    """

    def __init__(self, img_channels, text_embedding_dim, feature_maps=128):
        super().__init__()
        self.apply_bezier_activation = BezierActivation()

        # Progressive downsampling: 1024 → 512 → 256 → 128 → 64 → features
        self.initial_conv_1024_t = spectral_norm(
            nn.Conv2d(
                img_channels,
                max(img_channels, feature_maps // 16),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_1024_p0 = spectral_norm(
            nn.Conv2d(
                img_channels,
                max(img_channels, feature_maps // 16),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_1024_p1 = spectral_norm(
            nn.Conv2d(
                img_channels,
                max(img_channels, feature_maps // 16),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_1024_p2 = spectral_norm(
            nn.Conv2d(
                img_channels,
                max(img_channels, feature_maps // 16),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_1024_p3 = spectral_norm(
            nn.Conv2d(
                img_channels,
                max(img_channels, feature_maps // 16),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )

        self.initial_conv_512_t = spectral_norm(
            nn.Conv2d(
                max(img_channels, feature_maps // 16),
                max(img_channels, feature_maps // 8),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_512_p0 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 16),
                max(img_channels, feature_maps // 8),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_512_p1 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 16),
                max(img_channels, feature_maps // 8),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_512_p2 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 16),
                max(img_channels, feature_maps // 8),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_512_p3 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 16),
                max(img_channels, feature_maps // 8),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )

        self.initial_conv_256_t = spectral_norm(
            nn.Conv2d(
                max(img_channels, feature_maps // 8),
                max(img_channels, feature_maps // 4),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_256_p0 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 8),
                max(img_channels, feature_maps // 4),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_256_p1 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 8),
                max(img_channels, feature_maps // 4),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_256_p2 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 8),
                max(img_channels, feature_maps // 4),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_256_p3 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 8),
                max(img_channels, feature_maps // 4),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )

        self.initial_conv_128_t = spectral_norm(
            nn.Conv2d(
                max(img_channels, feature_maps // 4),
                max(img_channels, feature_maps // 2),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_128_p0 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 4),
                max(img_channels, feature_maps // 2),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_128_p1 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 4),
                max(img_channels, feature_maps // 2),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_128_p2 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 4),
                max(img_channels, feature_maps // 2),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )
        self.initial_conv_128_p3 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 4),
                max(img_channels, feature_maps // 2),
                kernel_size=6,
                stride=2,
                padding=2,
            )
        )

        self.initial_conv_feat_t = spectral_norm(
            nn.Conv2d(
                max(img_channels, feature_maps // 2),
                feature_maps,
                kernel_size=7,
                padding=3,
            )
        )
        self.initial_conv_feat_p0 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 2),
                feature_maps,
                kernel_size=7,
                padding=3,
            )
        )
        self.initial_conv_feat_p1 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 2),
                feature_maps,
                kernel_size=7,
                padding=3,
            )
        )
        self.initial_conv_feat_p2 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 2),
                feature_maps,
                kernel_size=7,
                padding=3,
            )
        )
        self.initial_conv_feat_p3 = spectral_norm(
            nn.Conv2d(
                2 * max(img_channels, feature_maps // 2),
                feature_maps,
                kernel_size=7,
                padding=3,
            )
        )

        # Context and embedding modules
        self.context_module = LeanContext2D(
            in_channels=feature_maps,
            out_channels=feature_maps,
            num_heads=max(1, feature_maps // 16),
        )

        self.validity_context_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 5, kernel_size=3, padding=0)),
            self.apply_bezier_activation,
        )
        self.spade = SPADE(context_nc=feature_maps, num_features=feature_maps)

        self.matching_head = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    feature_maps,
                    text_embedding_dim * 5,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            ),
            self.apply_bezier_activation,
        )
        self.apply(xavier_init)

    def downsample(self, x):
        """Progressive downsampling with Bezier activations."""
        # 1024 → 512
        p0 = self.initial_conv_1024_p0(x)
        p1 = self.initial_conv_1024_p1(x)
        p2 = self.initial_conv_1024_p2(x)
        p3 = self.initial_conv_1024_p3(x)
        x = self.initial_conv_1024_t(x)
        x = self.apply_bezier_activation(torch.cat([x, p0, p1, p2, p3], dim=1))

        # 512 → 256
        p0 = self.initial_conv_512_p0(torch.cat([x, p0], dim=1))
        p1 = self.initial_conv_512_p1(torch.cat([x, p1], dim=1))
        p2 = self.initial_conv_512_p2(torch.cat([x, p2], dim=1))
        p3 = self.initial_conv_512_p3(torch.cat([x, p3], dim=1))
        x = self.initial_conv_512_t(x)
        x = self.apply_bezier_activation(torch.cat([x, p0, p1, p2, p3], dim=1))

        # 256 → 128
        p0 = self.initial_conv_256_p0(torch.cat([x, p0], dim=1))
        p1 = self.initial_conv_256_p1(torch.cat([x, p1], dim=1))
        p2 = self.initial_conv_256_p2(torch.cat([x, p2], dim=1))
        p3 = self.initial_conv_256_p3(torch.cat([x, p3], dim=1))
        x = self.initial_conv_256_t(x)
        x = self.apply_bezier_activation(torch.cat([x, p0, p1, p2, p3], dim=1))

        # 128 → 64
        p0 = self.initial_conv_128_p0(torch.cat([x, p0], dim=1))
        p1 = self.initial_conv_128_p1(torch.cat([x, p1], dim=1))
        p2 = self.initial_conv_128_p2(torch.cat([x, p2], dim=1))
        p3 = self.initial_conv_128_p3(torch.cat([x, p3], dim=1))
        x = self.initial_conv_128_t(x)
        x = self.apply_bezier_activation(torch.cat([x, p0, p1, p2, p3], dim=1))

        # 64 → feature_maps
        p0 = self.initial_conv_feat_p0(torch.cat([x, p0], dim=1))
        p1 = self.initial_conv_feat_p1(torch.cat([x, p1], dim=1))
        p2 = self.initial_conv_feat_p2(torch.cat([x, p2], dim=1))
        p3 = self.initial_conv_feat_p3(torch.cat([x, p3], dim=1))
        x = self.initial_conv_feat_t(x)
        x = self.apply_bezier_activation(torch.cat([x, p0, p1, p2, p3], dim=1))

        return x

    def build_embed_validation(self, x, images):
        """Build image embeddings from downsampled features."""
        B = x.size(0)
        # Generate context & transform
        context = self.context_module(x)  # [B, feature_maps, H', W']
        x = self.validity_context_layer(x)  # [B, feature_maps*5, H'-2, W'-2]
        x = self.spade(x, context)  # [B, feature_maps*5, H'-2, W'-2]

        x = torch.mean(x, dim=[2, 3])  # [B, feature_maps*5]
        x = x.view(B, -1, 1, 1)
        x = self.matching_head(x).view(images.size(0), -1)  # [B, text_embedding_dim]
        return x

    def forward(self, images):
        """
        Args:
            images: Input images [B, C, H, W]

        Returns:
            Image embeddings [B, text_embedding_dim]
        """
        # Get initial features
        downsampled = self.downsample(images)
        embed = self.build_embed_validation(downsampled, images)

        return embed
