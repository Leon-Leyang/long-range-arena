# Copyright 2021 Google LLC
# Updated for modern Flax (Linen) API and Optax

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
"""Main script for document matching - Updated for modern Flax/Optax."""

import functools
import itertools
import json
import os
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax.training import train_state
from flax.training import common_utils
import orbax.checkpoint as ocp
import jax
from jax import random
import jax.numpy as jnp
import optax
from lra_benchmarks.matching import input_pipeline_pickle as input_pipeline
from lra_benchmarks.utils import train_utils
from lra_benchmarks.models.transformer import transformer
from ml_collections import config_flags
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_string(
    'task_name', default='aan', help='Directory to store model data.')
flags.DEFINE_string(
    'data_dir', default=None, help='Directory containing datasets.')
flags.DEFINE_string(
    'vocab_file_path', default=None, help='Path for vocab file.')
flags.DEFINE_bool(
    'test_only', default=False, help='Run the evaluation on the test data.')


class TrainState(train_state.TrainState):
  """Custom train state with dropout key."""
  dropout_rng: Any = None


def create_train_state(rng, model, learning_rate, weight_decay, input_shape):
  """Creates initial TrainState for dual encoder."""
  dropout_rng, params_rng = random.split(rng)
  
  # Initialize model with two inputs
  dummy_input1 = jnp.ones(input_shape, dtype=jnp.int32)
  dummy_input2 = jnp.ones(input_shape, dtype=jnp.int32)
  variables = model.init({'params': params_rng, 'dropout': dropout_rng}, 
                         dummy_input1, dummy_input2, train=False)
  params = variables['params']
  
  # Create optimizer with AdamW
  tx = optax.adamw(
      learning_rate=learning_rate,
      b1=0.9,
      b2=0.98,
      eps=1e-9,
      weight_decay=weight_decay
  )
  
  return TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      dropout_rng=dropout_rng,
  )


def compute_metrics(logits, labels, num_classes, weights):
  """Compute summary metrics."""
  loss, weight_sum = train_utils.compute_weighted_cross_entropy(
      logits, labels, num_classes=num_classes, weights=weights)
  acc, _ = train_utils.compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  metrics = jax.lax.psum(metrics, 'batch')
  return metrics


def train_step(state, batch, learning_rate_fn, num_classes):
  """Perform a single training step."""
  inputs1 = batch['inputs1']
  inputs2 = batch['inputs2']
  targets = batch['targets']

  # Split dropout key
  dropout_rng, new_dropout_rng = random.split(state.dropout_rng)

  def loss_fn(params):
    """Loss function used for training."""
    logits = state.apply_fn(
        {'params': params}, 
        inputs1, inputs2,
        train=True,
        rngs={'dropout': dropout_rng})
    loss, weight_sum = train_utils.compute_weighted_cross_entropy(
        logits, targets, num_classes=num_classes, weights=None)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, 'batch')
  
  state = state.apply_gradients(grads=grads)
  state = state.replace(dropout_rng=new_dropout_rng)
  
  metrics = compute_metrics(logits, targets, num_classes, None)
  metrics['learning_rate'] = lr

  return state, metrics


def eval_step(state, batch, num_classes):
  """Evaluation step."""
  inputs1 = batch['inputs1']
  inputs2 = batch['inputs2']
  targets = batch['targets']
  logits = state.apply_fn({'params': state.params}, inputs1, inputs2, train=False)
  return compute_metrics(logits, targets, num_classes, None)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = FLAGS.config
  logging.info('===========Config Dict============')
  logging.info(config)
  batch_size = config.batch_size
  learning_rate = config.learning_rate
  num_train_steps = config.num_train_steps
  num_eval_steps = config.num_eval_steps
  eval_freq = config.eval_frequency
  random_seed = config.random_seed
  model_type = config.model_type
  num_classes = config.num_classes

  if jax.process_index() == 0:
    summary_writer = SummaryWriter(os.path.join(FLAGS.model_dir, 'summary'))

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  train_ds, eval_ds, test_ds, encoder = input_pipeline.get_matching_datasets(
      n_devices=jax.local_device_count(),
      task_name=FLAGS.task_name,
      data_dir=FLAGS.data_dir,
      batch_size=batch_size,
      max_length=config.max_length,
      vocab_file_path=FLAGS.vocab_file_path)

  vocab_size = encoder.vocab_size
  logging.info('Vocab Size: %d', vocab_size)

  train_ds = train_ds.repeat()
  train_iter = iter(train_ds)
  max_length = config.max_length
  input_shape = (batch_size, max_length)

  # Use dual encoder for matching tasks
  model_kwargs = {
      'vocab_size': vocab_size,
      'emb_dim': config.emb_dim,
      'num_heads': config.num_heads,
      'num_layers': config.num_layers,
      'qkv_dim': config.qkv_dim,
      'mlp_dim': config.mlp_dim,
      'max_len': config.max_length,
      'classifier': True,
      'num_classes': num_classes,
  }

  rng = random.PRNGKey(random_seed)
  rng = jax.random.fold_in(rng, jax.process_index())
  rng, init_rng = random.split(rng)

  # Create dual encoder model
  model = transformer.TransformerDualEncoder(**model_kwargs)
  
  # Create train state
  state = create_train_state(
      init_rng, model, learning_rate, 
      config.weight_decay, input_shape)
  
  start_step = 0
  
  # Checkpoint management
  ckpt_dir = os.path.join(FLAGS.model_dir, 'checkpoints')
  os.makedirs(ckpt_dir, exist_ok=True)
  
  if config.restore_checkpoints or FLAGS.test_only:
    checkpointer = ocp.StandardCheckpointer()
    if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
      latest_step = max([int(d.split('_')[-1]) for d in os.listdir(ckpt_dir) 
                        if d.startswith('checkpoint_')])
      ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{latest_step}')
      restored = checkpointer.restore(ckpt_path, state)
      state = restored
      start_step = int(state.step)
      logging.info('Restored checkpoint from step %d', start_step)

  # Replicate state
  state = jax_utils.replicate(state)

  learning_rate_fn = train_utils.create_learning_rate_scheduler(
      base_learning_rate=learning_rate)
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn,
                        num_classes=num_classes),
      axis_name='batch')
  p_eval_step = jax.pmap(
      functools.partial(eval_step, num_classes=num_classes),
      axis_name='batch')

  def run_eval(eval_ds, num_eval_steps=-1):
    eval_metrics = []
    eval_iter = iter(eval_ds)
    if num_eval_steps == -1:
      num_iter = itertools.count()
    else:
      num_iter = range(num_eval_steps)
    for _, eval_batch in zip(num_iter, eval_iter):
      eval_batch = {k: v.numpy() for k, v in eval_batch.items()}
      eval_batch = common_utils.shard(eval_batch)
      metrics = p_eval_step(state, eval_batch)
      eval_metrics.append(metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics_sums = jax.tree_util.tree_map(jnp.sum, eval_metrics)
    eval_denominator = eval_metrics_sums.pop('denominator')
    eval_summary = jax.tree_util.tree_map(
        lambda x: x / eval_denominator,
        eval_metrics_sums)
    eval_summary['perplexity'] = jnp.clip(
        jnp.exp(eval_summary['loss']), a_max=1.0e4)
    return eval_summary

  if FLAGS.test_only:
    with tf.io.gfile.GFile(os.path.join(FLAGS.model_dir, 'results.json'),
                           'w') as f:
      test_summary = run_eval(test_ds)
      json.dump(jax.tree_util.tree_map(lambda x: x.tolist(), test_summary), f)
    return

  metrics_all = []
  tick = time.time()
  for step, batch in zip(range(start_step, num_train_steps), train_iter):
    batch = {k: v.numpy() for k, v in batch.items()}
    batch = common_utils.shard(batch)
    state, metrics = p_train_step(state, batch)
    metrics_all.append(metrics)
    logging.info('train in step: %d', step)

    # Save a Checkpoint
    if ((step % config.checkpoint_freq == 0 and step > 0) or
        step == num_train_steps - 1):
      if jax.process_index() == 0 and config.save_checkpoints:
        unreplicated_state = jax_utils.unreplicate(state)
        checkpointer = ocp.StandardCheckpointer()
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{step}')
        checkpointer.save(ckpt_path, unreplicated_state)

    # Periodic metric handling.
    if step % eval_freq == 0 and step > 0:
      metrics_all = common_utils.get_metrics(metrics_all)
      lr = metrics_all.pop('learning_rate').mean()
      metrics_sums = jax.tree_util.tree_map(jnp.sum, metrics_all)
      denominator = metrics_sums.pop('denominator')
      summary = jax.tree_util.tree_map(lambda x: x / denominator, metrics_sums)
      summary['learning_rate'] = lr
      summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)
      logging.info('train in step: %d, loss: %.4f', step, summary['loss'])
      if jax.process_index() == 0:
        tock = time.time()
        steps_per_sec = eval_freq / (tock - tick)
        tick = tock
        summary_writer.add_scalar('steps per second', steps_per_sec, step)
        for key, val in summary.items():
          summary_writer.add_scalar(f'train_{key}', float(val), step)
        summary_writer.flush()
      metrics_all = []

      # Eval Metrics
      eval_summary = run_eval(eval_ds, num_eval_steps)
      logging.info('eval in step: %d, loss: %.4f, acc: %.4f', step,
                   eval_summary['loss'], eval_summary['accuracy'])
      if jax.process_index() == 0:
        for key, val in eval_summary.items():
          summary_writer.add_scalar(f'eval_{key}', float(val), step)
        summary_writer.flush()


if __name__ == '__main__':
  app.run(main)
