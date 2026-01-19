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
"""Main training script for image classification - Updated for modern Flax/Optax."""

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
from lra_benchmarks.image import task_registry
from lra_benchmarks.utils import train_utils
from ml_collections import config_flags
import tensorflow as tf
from tensorboardX import SummaryWriter


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_string('task_name', default='mnist', help='Name of the task')
flags.DEFINE_bool(
    'eval_only', default=False, help='Run the evaluation on the test data.')


class TrainState(train_state.TrainState):
  """Custom train state with dropout key."""
  dropout_rng: Any = None


def create_train_state(rng, model, learning_rate, weight_decay, input_shape):
  """Creates initial TrainState."""
  dropout_rng, params_rng = random.split(rng)
  
  # Initialize model
  dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
  variables = model.init({'params': params_rng, 'dropout': dropout_rng}, 
                         dummy_input, train=False)
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
      logits, labels, num_classes, weights=weights)
  acc, _ = train_utils.compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  metrics = jax.lax.psum(metrics, 'batch')
  return metrics


def train_step(state, batch, learning_rate_fn, num_classes, 
               flatten_input=True, grad_clip_norm=None):
  """Perform a single training step."""
  inputs = batch['inputs']
  targets = batch['targets']
  
  if flatten_input:
    inputs = inputs.reshape(inputs.shape[0], -1)

  # Split dropout key
  dropout_rng, new_dropout_rng = random.split(state.dropout_rng)

  def loss_fn(params):
    """Loss function used for training."""
    logits = state.apply_fn(
        {'params': params}, 
        inputs, 
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
  
  # Optionally clip gradients
  if grad_clip_norm is not None:
    grads_flat, tree_def = jax.tree_util.tree_flatten(grads)
    g_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
    g_factor = jnp.minimum(1.0, grad_clip_norm / g_l2)
    grads = jax.tree_util.tree_map(lambda p: g_factor * p, grads)
  
  state = state.apply_gradients(grads=grads)
  state = state.replace(dropout_rng=new_dropout_rng)
  
  metrics = compute_metrics(logits, targets, num_classes, weights=None)
  metrics['learning_rate'] = lr

  return state, metrics


def eval_step(state, batch, num_classes, flatten_input=True):
  """Evaluation step."""
  inputs = batch['inputs']
  targets = batch['targets']
  
  if flatten_input:
    inputs = inputs.reshape(inputs.shape[0], -1)
    
  logits = state.apply_fn({'params': state.params}, inputs, train=False)
  return compute_metrics(logits, targets, num_classes, weights=None)


def test(state, p_eval_step, step, test_ds, summary_writer, model_dir):
  """Test the model on test_ds."""
  test_metrics = []
  test_iter = iter(test_ds)
  for _, test_batch in zip(itertools.repeat(1), test_iter):
    test_batch = {k: v.numpy() for k, v in test_batch.items()}
    test_batch = common_utils.shard(test_batch)
    metrics = p_eval_step(state, test_batch)
    test_metrics.append(metrics)
  test_metrics = common_utils.get_metrics(test_metrics)
  test_metrics_sums = jax.tree_util.tree_map(jnp.sum, test_metrics)
  test_denominator = test_metrics_sums.pop('denominator')
  test_summary = jax.tree_util.tree_map(
      lambda x: x / test_denominator,
      test_metrics_sums)
  logging.info('test in step: %d, loss: %.4f, acc: %.4f', step,
               test_summary['loss'], test_summary['accuracy'])
  if jax.process_index() == 0:
    for key, val in test_summary.items():
      summary_writer.add_scalar(f'test_{key}', float(val), step)
    summary_writer.flush()
  with tf.io.gfile.GFile(os.path.join(model_dir, 'results.json'), 'w') as f:
    json.dump(jax.tree_util.tree_map(lambda x: x.tolist(), test_summary), f)


def train_loop(config, state, eval_ds, eval_freq, num_eval_steps,
               num_train_steps, p_eval_step, p_train_step,
               start_step, train_iter, summary_writer, ckpt_dir):
  """Training loop."""
  metrics_all = []
  tick = time.time()
  logging.info('Starting training')
  logging.info('====================')

  step = start_step
  for step, batch in zip(range(start_step, num_train_steps), train_iter):
    batch = {k: v.numpy() for k, v in batch.items()}
    batch = common_utils.shard(batch)
    state, metrics = p_train_step(state, batch)
    metrics_all.append(metrics)
    
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
      logging.info('train in step: %d, loss: %.4f, acc: %.4f', step,
                   summary['loss'], summary['accuracy'])
      if jax.process_index() == 0:
        tock = time.time()
        steps_per_sec = eval_freq / (tock - tick)
        tick = tock
        summary_writer.add_scalar('examples_per_second',
                                  steps_per_sec * config.batch_size, step)
        for key, val in summary.items():
          summary_writer.add_scalar(f'train_{key}', float(val), step)
        summary_writer.flush()
      metrics_all = []

      # Eval Metrics
      eval_metrics = []
      eval_iter = iter(eval_ds)
      if num_eval_steps == -1:
        num_iter = itertools.repeat(1)
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
      logging.info('eval in step: %d, loss: %.4f, acc: %.4f', step,
                   eval_summary['loss'], eval_summary['accuracy'])
      if jax.process_index() == 0:
        for key, val in eval_summary.items():
          summary_writer.add_scalar(f'val_{key}', float(val), step)
        summary_writer.flush()
  return state, step


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

  if jax.process_index() == 0:
    summary_writer = SummaryWriter(os.path.join(FLAGS.model_dir, 'summary'))
  else:
    summary_writer = None

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  logging.info('Training on %s', FLAGS.task_name)

  if model_type in ['wideresnet', 'resnet', 'simple_cnn']:
    normalize = True
  else:  # transformer-based models
    normalize = False
  (train_ds, eval_ds, test_ds, num_classes, vocab_size,
   input_shape) = task_registry.TASK_DATA_DICT[FLAGS.task_name](
       n_devices=jax.local_device_count(),
       batch_size=batch_size,
       normalize=normalize)
  train_ds = train_ds.repeat()
  train_iter = iter(train_ds)
  
  model_kwargs = {}
  flatten_input = True

  if model_type in ['wideresnet', 'resnet', 'simple_cnn']:
    model_kwargs.update({
        'num_classes': num_classes,
    })
    flatten_input = False
  else:  # transformer models
    # we will flatten the input
    bs, h, w, c = input_shape
    assert c == 1
    input_shape = (bs, h * w * c)
    model_kwargs.update({
        'vocab_size': vocab_size,
        'max_len': input_shape[1],
        'classifier': True,
        'num_classes': num_classes,
    })

  model_kwargs.update(config.model)

  rng = random.PRNGKey(random_seed)
  rng = jax.random.fold_in(rng, jax.process_index())
  rng, init_rng = random.split(rng)

  # Create model
  model_class = train_utils.get_model_class(model_type)
  model = model_class(**model_kwargs)
  
  # Create train state
  state = create_train_state(
      init_rng, model, learning_rate, 
      config.weight_decay, input_shape)

  start_step = 0
  
  # Checkpoint management
  ckpt_dir = os.path.join(FLAGS.model_dir, 'checkpoints')
  os.makedirs(ckpt_dir, exist_ok=True)
  
  if config.restore_checkpoints:
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
      factors=config.factors,
      base_learning_rate=learning_rate,
      warmup_steps=config.warmup,
      steps_per_cycle=config.get('steps_per_cycle', None),
  )
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          learning_rate_fn=learning_rate_fn,
          num_classes=num_classes,
          grad_clip_norm=config.get('grad_clip_norm', None),
          flatten_input=flatten_input),
      axis_name='batch')

  p_eval_step = jax.pmap(
      functools.partial(
          eval_step, num_classes=num_classes, flatten_input=flatten_input),
      axis_name='batch',
  )

  state, step = train_loop(config, state, eval_ds, eval_freq,
                           num_eval_steps, num_train_steps,
                           p_eval_step, p_train_step, start_step, train_iter,
                           summary_writer, ckpt_dir)

  logging.info('Starting testing')
  logging.info('====================')
  test(state, p_eval_step, step, test_ds, summary_writer, FLAGS.model_dir)


if __name__ == '__main__':
  app.run(main)
