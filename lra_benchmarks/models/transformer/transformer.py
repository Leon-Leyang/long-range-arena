# Copyright 2021 Google LLC
# Updated for modern Flax (Linen) API

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer model - Updated for Flax Linen API."""

from flax import linen as nn
import jax.numpy as jnp
from lra_benchmarks.models.layers import common_layers
from typing import Optional, Callable


class TransformerBlock(nn.Module):
  """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""
  qkv_dim: int
  mlp_dim: int
  num_heads: int
  dtype: jnp.dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self,
               inputs,
               inputs_segmentation=None,
               padding_mask=None,
               deterministic: bool = False):
    """Applies TransformerBlock module.

    Args:
      inputs: input data
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: bool, mask padding tokens
      deterministic: bool, deterministic or not (to apply dropout)

    Returns:
      output after transformer block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    
    # Apply attention - Flax's MultiHeadDotProductAttention doesn't support padding_mask
    # directly, so we mask the output after attention computation
    # This is more memory efficient than creating a full [seq_len, seq_len] attention mask
    x_attn = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=deterministic)(x, x)
    
    # Mask out padded positions in the output (zero them out)
    if padding_mask is not None:
      x_attn = x_attn * padding_mask
    
    x = nn.Dropout(rate=self.dropout_rate)(x_attn, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = common_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate)(y, deterministic=deterministic)

    return x + y


class TransformerEncoder(nn.Module):
  """Transformer Model Encoder."""
  vocab_size: int
  emb_dim: int = 512
  num_heads: int = 8
  dtype: jnp.dtype = jnp.float32
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 512
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  learn_pos_emb: bool = False
  classifier: bool = False
  classifier_pool: str = 'CLS'
  num_classes: int = 10

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               inputs_segmentation=None,
               train: bool = True):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      train: if it is training

    Returns:
      output of a transformer encoder or logits if classifier_mode is true.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[..., None]

    # Input Embedding
    x = inputs.astype('int32')
    x = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0))(x)

    max_len = self.max_len
    if self.classifier and self.classifier_pool == 'CLS':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, self.emb_dim))
      cls = jnp.tile(cls, [x.shape[0], 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
      max_len += 1
      src_padding_mask = jnp.concatenate(
          [src_padding_mask[:, :1], src_padding_mask], axis=1)

    pe_init = nn.initializers.normal(stddev=0.02) if self.learn_pos_emb else None
    x = common_layers.AddPositionEmbs(
        max_len=max_len,
        posemb_init=pe_init,
        name='posembed_input')(x, inputs_positions=inputs_positions)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    dtype = self.dtype

    # Input Encoder
    for lyr in range(self.num_layers):
      x = TransformerBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}')(
              x,
              padding_mask=src_padding_mask,
              inputs_segmentation=inputs_segmentation,
              deterministic=not train)

    encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)

    if self.classifier:
      encoded = common_layers.ClassifierHead(
          num_classes=self.num_classes,
          mlp_dim=self.mlp_dim,
          pooling_mode=self.classifier_pool)(encoded)
    return encoded


class TransformerDualEncoder(nn.Module):
  """Transformer Model for Matching (dual encoding) tasks."""
  vocab_size: int
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  classifier: bool = True
  classifier_pool: str = 'CLS'
  num_classes: int = 2
  interaction: Optional[str] = None

  @nn.compact
  def __call__(self,
               inputs1,
               inputs2,
               inputs1_positions=None,
               inputs2_positions=None,
               inputs1_segmentation=None,
               inputs2_segmentation=None,
               train: bool = False):
    """Applies Transformer model on text similarity.

    Args:
      inputs1: input data.
      inputs2: target data.
      inputs1_positions: input subsequence positions for packed examples.
      inputs2_positions: target subsequence positions for packed examples.
      inputs1_segmentation: input segmentation info for packed examples.
      inputs2_segmentation: target segmentation info for packed examples.
      train: whether it is training.

    Returns:
      output of a transformer decoder.
    """
    # Shared encoder
    encoder = TransformerEncoder(
        vocab_size=self.vocab_size,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        num_layers=self.num_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        max_len=self.max_len,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        classifier=False,  # Don't classify yet
        name='encoder')

    inputs1_encoded = encoder(
        inputs=inputs1,
        inputs_positions=inputs1_positions,
        inputs_segmentation=inputs1_segmentation,
        train=train)
    inputs2_encoded = encoder(
        inputs=inputs2,
        inputs_positions=inputs2_positions,
        inputs_segmentation=inputs2_segmentation,
        train=train)

    encoded = common_layers.ClassifierHeadDual(
        num_classes=self.num_classes,
        mlp_dim=self.mlp_dim,
        pooling_mode=self.classifier_pool,
        interaction=self.interaction)(inputs1_encoded, inputs2_encoded)

    return encoded
