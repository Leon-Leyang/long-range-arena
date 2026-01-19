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
"""Input pipeline for image datasets using pickle files."""

import pickle
from pathlib import Path
import numpy as np
import tensorflow.compat.v1 as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


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
    elif 'input_ids_0' in data and 'label' in data:
      # Kaggle format for image datasets
      inputs = data['input_ids_0']
      targets = data['label']
    else:
      raise ValueError(f"Unexpected dict keys: {list(data.keys())}")
  elif isinstance(data, (list, tuple)):
    # List of tuples or list of dicts
    if len(data) == 0:
      return [], []
    
    first_item = data[0]
    if isinstance(first_item, dict):
      # List of dictionaries - try multiple key patterns
      input_key = None
      label_key = None
      
      # Find input key
      for key in ['input_ids_0', 'data', 'inputs', 'input_ids', 'x']:
        if key in first_item:
          input_key = key
          break
      
      # Find label key
      for key in ['label', 'labels', 'targets', 'y']:
        if key in first_item:
          label_key = key
          break
      
      if input_key and label_key:
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


def reshape_image_data(inputs, resolution=None):
  """Reshape image data to proper format.
  
  Images in pickle files may be:
  - Flattened: [batch, height*width] or [batch, height*width*channels]
  - 2D: [batch, height*width, channels]
  - 4D: [batch, height, width, channels]
  
  We need to convert to [batch, height*width, channels] for LRA.
  """
  inputs = np.array(inputs)
  
  if len(inputs.shape) == 2:
    # Flattened: [batch, height*width] or [batch, height*width*channels]
    if resolution is None:
      # Infer resolution from sequence length
      seq_len = inputs.shape[1]
      # Try common resolutions
      if seq_len == 1024:  # 32x32
        resolution = 32
      elif seq_len == 4096:  # 64x64
        resolution = 64
      elif seq_len == 16384:  # 128x128
        resolution = 128
      elif seq_len == 65536:  # 256x256
        resolution = 256
      else:
        # Assume 1 channel
        resolution = int(np.sqrt(seq_len))
    
    # Reshape to [batch, height*width, 1] (grayscale)
    batch_size = inputs.shape[0]
    inputs = inputs.reshape(batch_size, resolution * resolution, 1)
    
  elif len(inputs.shape) == 3:
    # Already in [batch, height*width, channels] format
    if inputs.shape[2] > 1:
      # Convert RGB to grayscale if needed
      # Simple average (can be improved)
      inputs = np.mean(inputs, axis=2, keepdims=True)
    elif inputs.shape[2] == 1:
      # Already grayscale
      pass
    else:
      # Add channel dimension
      inputs = np.expand_dims(inputs, axis=2)
      
  elif len(inputs.shape) == 4:
    # 4D: [batch, height, width, channels]
    batch_size, h, w, c = inputs.shape
    if c > 1:
      # Convert RGB to grayscale
      inputs = np.mean(inputs, axis=3, keepdims=True)
    # Flatten spatial dimensions
    inputs = inputs.reshape(batch_size, h * w, 1)
  
  return inputs


def get_cifar10_datasets(n_devices, batch_size=256, normalize=False, data_dir=None):
  """Get CIFAR-10 dataset splits from pickle files."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  # Determine pickle file paths
  if data_dir is None:
    pickle_dir = Path('./data')
  else:
    pickle_dir = Path(data_dir)
  
  file_prefix = 'lra-image'
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

  # Reshape image data
  train_inputs = reshape_image_data(train_inputs, resolution=32)
  val_inputs = reshape_image_data(val_inputs, resolution=32)
  test_inputs = reshape_image_data(test_inputs, resolution=32)

  # Convert to numpy arrays and ensure correct types
  train_inputs = np.array(train_inputs, dtype=np.int32)
  train_targets = np.array(train_targets, dtype=np.int32)
  val_inputs = np.array(val_inputs, dtype=np.int32)
  val_targets = np.array(val_targets, dtype=np.int32)
  test_inputs = np.array(test_inputs, dtype=np.int32)
  test_targets = np.array(test_targets, dtype=np.int32)

  # Normalize if requested
  if normalize:
    train_inputs = train_inputs / 255.0
    val_inputs = val_inputs / 255.0
    test_inputs = test_inputs / 255.0
    train_inputs = train_inputs.astype(np.float32)
    val_inputs = val_inputs.astype(np.float32)
    test_inputs = test_inputs.astype(np.float32)

  # Create TensorFlow datasets
  def create_dataset(inputs, targets, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices({
        'inputs': tf.constant(inputs, dtype=tf.int32 if not normalize else tf.float32),
        'targets': tf.constant(targets, dtype=tf.int32)
    })
    
    if shuffle:
      dataset = dataset.shuffle(
          buffer_size=min(256, len(inputs)), 
          reshuffle_each_iteration=True)
    
    # Batch (data is already padded/fixed size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

  train_dataset = create_dataset(train_inputs, train_targets, shuffle=True)
  val_dataset = create_dataset(val_inputs, val_targets, shuffle=False)
  test_dataset = create_dataset(test_inputs, test_targets, shuffle=False)

  return train_dataset, val_dataset, test_dataset, 10, 256, (batch_size, 32, 32, 1)


def get_pathfinder_base_datasets(n_devices,
                                 batch_size=256,
                                 resolution=32,
                                 normalize=False,
                                 split='easy',
                                 data_dir=None):
  """Get Pathfinder dataset splits from pickle files."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  if split not in ['easy', 'intermediate', 'hard']:
    raise ValueError("split must be in ['easy', 'intermediate', 'hard'].")

  # Determine pickle file paths
  if data_dir is None:
    pickle_dir = Path('./data')
  else:
    pickle_dir = Path(data_dir)
  
  # Map resolution to file prefix
  if resolution == 32:
    file_prefix = 'lra-pathfinder32-curv_contour_length_14'
    inputs_shape = (batch_size, 32, 32, 1)
  elif resolution == 64:
    file_prefix = 'lra-pathfinder64-curv_contour_length_14'
    inputs_shape = (batch_size, 64, 64, 1)
  elif resolution == 128:
    file_prefix = 'lra-pathfinder128-curv_contour_length_14'
    inputs_shape = (batch_size, 128, 128, 1)
  elif resolution == 256:
    file_prefix = 'lra-pathfinder256-curv_contour_length_14'
    inputs_shape = (batch_size, 256, 256, 1)
  else:
    raise ValueError('Resolution must be in [32, 64, 128, 256].')

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

  # Reshape image data
  train_inputs = reshape_image_data(train_inputs, resolution=resolution)
  val_inputs = reshape_image_data(val_inputs, resolution=resolution)
  test_inputs = reshape_image_data(test_inputs, resolution=resolution)

  # Convert to numpy arrays and ensure correct types
  train_inputs = np.array(train_inputs, dtype=np.int32)
  train_targets = np.array(train_targets, dtype=np.int32)
  val_inputs = np.array(val_inputs, dtype=np.int32)
  val_targets = np.array(val_targets, dtype=np.int32)
  test_inputs = np.array(test_inputs, dtype=np.int32)
  test_targets = np.array(test_targets, dtype=np.int32)

  # Normalize if requested
  if normalize:
    train_inputs = train_inputs / 255.0
    val_inputs = val_inputs / 255.0
    test_inputs = test_inputs / 255.0
    train_inputs = train_inputs.astype(np.float32)
    val_inputs = val_inputs.astype(np.float32)
    test_inputs = test_inputs.astype(np.float32)

  # Create TensorFlow datasets
  def create_dataset(inputs, targets, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices({
        'inputs': tf.constant(inputs, dtype=tf.int32 if not normalize else tf.float32),
        'targets': tf.constant(targets, dtype=tf.int32)
    })
    
    if shuffle:
      dataset = dataset.shuffle(
          buffer_size=min(256 * 8, len(inputs)), 
          reshuffle_each_iteration=True)
    
    # Batch (data is already padded/fixed size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

  train_dataset = create_dataset(train_inputs, train_targets, shuffle=True)
  val_dataset = create_dataset(val_inputs, val_targets, shuffle=False)
  test_dataset = create_dataset(test_inputs, test_targets, shuffle=False)

  return train_dataset, val_dataset, test_dataset, 2, 256, inputs_shape
