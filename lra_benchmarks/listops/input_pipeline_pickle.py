# Copyright 2021 Google LLC
#
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
"""Input pipeline for the listops dataset using pickle files."""

import pickle
from pathlib import Path
import numpy as np
import tensorflow.compat.v1 as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DummyEncoder:
  """Dummy encoder for compatibility - vocab size inferred from data."""
  
  def __init__(self, vocab_size):
    self.vocab_size = vocab_size


def load_pickle_data(pickle_path):
  """Load data from pickle file."""
  with open(pickle_path, 'rb') as f:
    data = pickle.load(f)
  return data


def extract_inputs_targets(data):
  """Extract inputs and targets from pickle data in various formats."""
  if isinstance(data, dict):
    # Dictionary format
    if 'inputs' in data and 'targets' in data:
      inputs = data['inputs']
      targets = data['targets']
    elif 'data' in data and 'labels' in data:
      inputs = data['data']
      targets = data['labels']
    elif 'input_ids' in data and 'label' in data:
      inputs = data['input_ids']
      targets = data['label']
    else:
      raise ValueError(f"Unexpected dict keys: {list(data.keys())}")
  elif isinstance(data, (list, tuple)):
    # List of tuples or list of dicts
    if len(data) == 0:
      return [], []
    
    first_item = data[0]
    if isinstance(first_item, dict):
      # List of dictionaries
      if 'input_ids' in first_item or 'inputs' in first_item:
        input_key = 'input_ids' if 'input_ids' in first_item else 'inputs'
        label_key = 'label' if 'label' in first_item else ('labels' if 'labels' in first_item else 'targets')
        inputs = [item[input_key] for item in data]
        targets = [item[label_key] for item in data]
      else:
        raise ValueError(f"Unexpected dict keys in sample: {list(first_item.keys())}")
    elif isinstance(first_item, (list, tuple)) and len(first_item) >= 2:
      # List of tuples: (input, label)
      inputs = [item[0] for item in data]
      targets = [item[1] for item in data]
    else:
      raise ValueError(f"Unexpected sample format: {type(first_item)}")
  else:
    raise ValueError(f"Unexpected data type: {type(data)}")
  
  return inputs, targets


def get_datasets(n_devices,
                 task_name,
                 data_dir=None,
                 batch_size=256,
                 max_length=2000):
  """Get listops datasets from pickle files."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  # Determine pickle file paths
  if data_dir is None:
    pickle_dir = Path('./data')
  else:
    pickle_dir = Path(data_dir)
  
  # Try different naming conventions
  file_prefix = 'lra-listops'
  train_path = pickle_dir / f"{file_prefix}.train.pickle"
  val_path = pickle_dir / f"{file_prefix}.dev.pickle"
  test_path = pickle_dir / f"{file_prefix}.test.pickle"
  
  # Fallback to val if dev doesn't exist
  if not val_path.exists():
    val_path = pickle_dir / f"{file_prefix}.val.pickle"

  if not train_path.exists():
    raise FileNotFoundError(f"Pickle file not found: {train_path}")
  if not val_path.exists():
    raise FileNotFoundError(f"Pickle file not found: {val_path}")
  if not test_path.exists():
    raise FileNotFoundError(f"Pickle file not found: {test_path}")

  tf.logging.info(f'Loading pickle files from {pickle_dir}')
  tf.logging.info(f'  Train: {train_path}')
  tf.logging.info(f'  Val: {val_path}')
  tf.logging.info(f'  Test: {test_path}')

  # Load pickle data
  train_data = load_pickle_data(train_path)
  val_data = load_pickle_data(val_path)
  test_data = load_pickle_data(test_path)

  # Extract inputs and targets
  train_inputs, train_targets = extract_inputs_targets(train_data)
  val_inputs, val_targets = extract_inputs_targets(val_data)
  test_inputs, test_targets = extract_inputs_targets(test_data)

  tf.logging.info(f'Loaded {len(train_inputs)} train samples')
  tf.logging.info(f'Loaded {len(val_inputs)} val samples')
  tf.logging.info(f'Loaded {len(test_inputs)} test samples')

  # Convert to numpy arrays and ensure correct types
  train_inputs = np.array(train_inputs, dtype=np.int32)
  train_targets = np.array(train_targets, dtype=np.int32)
  val_inputs = np.array(val_inputs, dtype=np.int32)
  val_targets = np.array(val_targets, dtype=np.int32)
  test_inputs = np.array(test_inputs, dtype=np.int32)
  test_targets = np.array(test_targets, dtype=np.int32)

  # Ensure inputs are padded to max_length (truncate if longer)
  if train_inputs.shape[1] > max_length:
    train_inputs = train_inputs[:, :max_length]
    val_inputs = val_inputs[:, :max_length]
    test_inputs = test_inputs[:, :max_length]
    tf.logging.info(f'Truncated sequences to max_length={max_length}')

  # Infer vocab size from data (max token value + 1)
  max_token = max(
      train_inputs.max() if len(train_inputs) > 0 else 0,
      val_inputs.max() if len(val_inputs) > 0 else 0,
      test_inputs.max() if len(test_inputs) > 0 else 0
  )
  vocab_size = int(max_token) + 1
  tf.logging.info(f'Inferred vocab_size={vocab_size} from data')

  # Create TensorFlow datasets
  def create_dataset(inputs, targets, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices({
        'inputs': tf.constant(inputs, dtype=tf.int32),
        'targets': tf.constant(targets, dtype=tf.int32)
    })
    
    if shuffle:
      dataset = dataset.shuffle(
          buffer_size=min(1024, len(inputs)), 
          reshuffle_each_iteration=True)
    
    # Batch (data is already padded)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

  train_dataset = create_dataset(train_inputs, train_targets, shuffle=True)
  val_dataset = create_dataset(val_inputs, val_targets, shuffle=False)
  test_dataset = create_dataset(test_inputs, test_targets, shuffle=False)

  # Create dummy encoder for compatibility
  encoder = DummyEncoder(vocab_size)

  return train_dataset, val_dataset, test_dataset, encoder
