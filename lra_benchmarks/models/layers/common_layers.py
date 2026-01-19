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
"""Common layers used in models - Updated for Flax Linen API."""

from flax import linen as nn
from jax import lax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


class Embed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """
  num_embeddings: int
  features: int
  mode: str = 'input'
  emb_init: Callable = nn.initializers.normal(stddev=1.0)

  @nn.compact
  def __call__(self, inputs):
    """Applies Embed module.

    Args:
      inputs: input data

    Returns:
      output which is embedded input data
    """
    embedding = self.param('embedding', self.emb_init,
                           (self.num_embeddings, self.features))
    if self.mode == 'input':
      if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError('Input type must be an integer or unsigned integer.')
      return jnp.take(embedding, inputs, axis=0)
    if self.mode == 'output':
      return jnp.einsum('bld,vd->blv', inputs, embedding)


def sinusoidal_init(max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs."""
  max_len: int = 512
  posemb_init: Optional[Callable] = None

  @nn.compact
  def __call__(self, inputs, inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, self.max_len, inputs.shape[-1])
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(
          max_len=self.max_len)(None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding', self.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP block."""
  mlp_dim: int
  dtype: jnp.dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(self.mlp_dim, dtype=self.dtype,
                 kernel_init=self.kernel_init, bias_init=self.bias_init)(inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim, dtype=self.dtype, kernel_init=self.kernel_init,
        bias_init=self.bias_init)(x)
    output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
    return output


class ClassifierHead(nn.Module):
  """Classifier head module."""
  num_classes: int
  mlp_dim: int
  pooling_mode: str = 'MEAN'

  @nn.compact
  def __call__(self, encoded):
    """Classifier head.

    Args:
      encoded: tensor inputs are shape of [bs, len, dim].

    Returns:
      tensor of shape [bs, num_classes]
    """
    if self.pooling_mode == 'MEAN':
      encoded = jnp.mean(encoded, axis=1)
    elif self.pooling_mode == 'SUM':
      encoded = jnp.sum(encoded, axis=1)
    elif self.pooling_mode == 'FLATTEN':
      encoded = encoded.reshape((encoded.shape[0], -1))
    elif self.pooling_mode == 'CLS':
      encoded = encoded[:, 0]
    else:
      raise NotImplementedError('Pooling not supported yet.')
    encoded = nn.Dense(self.mlp_dim, name='mlp')(encoded)
    encoded = nn.relu(encoded)
    encoded = nn.Dense(self.num_classes, name='logits')(encoded)
    return encoded


class ClassifierHeadDual(nn.Module):
  """Classifier head for dual encoding or pairwise problem."""
  num_classes: int
  mlp_dim: int
  pooling_mode: str = 'MEAN'
  interaction: Optional[str] = None

  @nn.compact
  def __call__(self, encoded1, encoded2):
    """Classifier head for dual encoding.

    Args:
      encoded1: tensor inputs are shape of [bs, len, dim].
      encoded2: tensor inputs are shape of [bs, len, dim].

    Returns:
      tensor of shape [bs, num_classes]
    """
    if self.pooling_mode == 'MEAN':
      encoded1 = jnp.mean(encoded1, axis=1)
      encoded2 = jnp.mean(encoded2, axis=1)
    elif self.pooling_mode == 'SUM':
      encoded1 = jnp.sum(encoded1, axis=1)
      encoded2 = jnp.sum(encoded2, axis=1)
    elif self.pooling_mode == 'FLATTEN':
      encoded1 = encoded1.reshape((encoded1.shape[0], -1))
      encoded2 = encoded2.reshape((encoded2.shape[0], -1))
    elif self.pooling_mode == 'CLS':
      encoded1 = encoded1[:, 0]
      encoded2 = encoded2[:, 0]
    else:
      raise NotImplementedError('Pooling not supported yet.')

    if self.interaction == 'NLI':
      # NLI interaction style
      encoded = jnp.concatenate([encoded1, encoded2, encoded1 * encoded2,
                                 encoded1 - encoded2], 1)
    else:
      encoded = jnp.concatenate([encoded1, encoded2], 1)
    encoded = nn.Dense(self.mlp_dim, name='mlp')(encoded)
    encoded = nn.relu(encoded)
    encoded = nn.Dense(int(self.mlp_dim // 2), name='mlp2')(encoded)
    encoded = nn.relu(encoded)
    encoded = nn.Dense(self.num_classes, name='logits')(encoded)
    return encoded


# Compatibility functions for old-style calls
def classifier_head(encoded, num_classes, mlp_dim, pooling_mode='MEAN'):
  """Functional wrapper for ClassifierHead - for backward compatibility."""
  return ClassifierHead(
      num_classes=num_classes,
      mlp_dim=mlp_dim,
      pooling_mode=pooling_mode
  )(encoded)


def classifier_head_dual(encoded1, encoded2, num_classes, mlp_dim,
                         pooling_mode='MEAN', interaction=None):
  """Functional wrapper for ClassifierHeadDual - for backward compatibility."""
  return ClassifierHeadDual(
      num_classes=num_classes,
      mlp_dim=mlp_dim,
      pooling_mode=pooling_mode,
      interaction=interaction
  )(encoded1, encoded2)
