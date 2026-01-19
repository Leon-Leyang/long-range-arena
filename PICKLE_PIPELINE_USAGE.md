# Using Pickle Input Pipelines

This directory contains pickle-based input pipelines that load preprocessed LRA datasets from pickle files instead of processing raw TSV/image files.

## Files Created

- `lra_benchmarks/listops/input_pipeline_pickle.py` - ListOps dataset loader
- `lra_benchmarks/text_classification/input_pipeline_pickle.py` - Text classification (IMDB) loader
- `lra_benchmarks/matching/input_pipeline_pickle.py` - Matching/retrieval (AAN) loader
- `lra_benchmarks/image/input_pipeline_pickle.py` - Image datasets (CIFAR, Pathfinder) loader

## Usage

### Option 1: Modify train.py to use pickle pipelines

Replace the import statement in your training scripts:

**Before:**
```python
from lra_benchmarks.listops import input_pipeline
```

**After:**
```python
from lra_benchmarks.listops import input_pipeline_pickle as input_pipeline
```

### Option 2: Add conditional loading

Modify the training scripts to check for pickle files first:

```python
import os
from pathlib import Path

# Check if pickle files exist
pickle_dir = Path('./data')
if (pickle_dir / 'lra-listops.train.pickle').exists():
    from lra_benchmarks.listops import input_pipeline_pickle as input_pipeline
    tf.logging.info('Using pickle input pipeline')
else:
    from lra_benchmarks.listops import input_pipeline
    tf.logging.info('Using original input pipeline')
```

### Option 3: Use environment variable

Modify training scripts to check an environment variable:

```python
import os

if os.environ.get('USE_PICKLE_DATA', 'false').lower() == 'true':
    from lra_benchmarks.listops import input_pipeline_pickle as input_pipeline
else:
    from lra_benchmarks.listops import input_pipeline
```

## Expected Pickle File Locations

The pickle pipelines expect files in the following locations (relative to `data_dir`):

### ListOps
- `lra-listops.train.pickle`
- `lra-listops.dev.pickle` or `lra-listops.val.pickle`
- `lra-listops.test.pickle`

### Text Classification (IMDB)
- `lra-text.train.pickle`
- `lra-text.dev.pickle` or `lra-text.val.pickle`
- `lra-text.test.pickle`

### Matching/Retrieval (AAN)
- `lra-retrieval.train.pickle`
- `lra-retrieval.dev.pickle` or `lra-retrieval.val.pickle` or `lra-retrieval.eval.pickle`
- `lra-retrieval.test.pickle`

### CIFAR-10
- `lra-image.train.pickle`
- `lra-image.dev.pickle` or `lra-image.val.pickle`
- `lra-image.test.pickle`

### Pathfinder
- `lra-pathfinder32-curv_contour_length_14.train.pickle`
- `lra-pathfinder32-curv_contour_length_14.dev.pickle` or `.val.pickle`
- `lra-pathfinder32-curv_contour_length_14.test.pickle`

(Similar for 64, 128, 256 resolutions)

## Pickle Data Format

The pickle files should contain preprocessed data in one of these formats:

### Text Datasets (ListOps, Text, Retrieval)

**Dictionary format:**
```python
{
    'inputs': [[token_ids...], ...],  # List of tokenized sequences
    'targets': [label1, label2, ...]   # List of labels
}
```

**List of tuples:**
```python
[
    ([token_ids...], label1),
    ([token_ids...], label2),
    ...
]
```

**List of dictionaries:**
```python
[
    {'input_ids': [token_ids...], 'label': label1},
    {'input_ids': [token_ids...], 'label': label2},
    ...
]
```

**For pair datasets (retrieval):**
```python
{
    'input_ids_0': [[token_ids...], ...],
    'input_ids_1': [[token_ids...], ...],
    'label': [label1, label2, ...]
}
```

### Image Datasets (CIFAR, Pathfinder)

**Dictionary format:**
```python
{
    'data': [[pixel_values...], ...],  # Flattened images
    'labels': [label1, label2, ...]
}
```

**List of tuples:**
```python
[
    ([pixel_values...], label1),
    ([pixel_values...], label2),
    ...
]
```

## Key Features

1. **Automatic format detection**: Handles multiple pickle data formats automatically
2. **Vocab size inference**: Automatically infers vocabulary size from token values
3. **Sequence truncation**: Truncates sequences to `max_length` if needed
4. **Image reshaping**: Automatically reshapes flattened images to proper format
5. **Compatibility**: Returns same format as original pipelines (TensorFlow datasets)

## Differences from Original Pipelines

1. **No preprocessing**: Pickle data is already tokenized, encoded, and padded
2. **No vocabulary building**: Vocab size is inferred from data
3. **Dummy encoder**: Returns a dummy encoder object for compatibility (vocab_size only)
4. **Faster loading**: Direct loading from pickle files is faster than processing TSV files

## Verification

Before using pickle pipelines, verify your pickle files:

```bash
python verify_lra_data.py --pickle-dir ./data --dataset listops
python verify_lra_data.py --pickle-dir ./data --dataset text
python verify_lra_data.py --pickle-dir ./data --dataset retrieval
python verify_lra_data.py --pickle-dir ./data --dataset image
python verify_lra_data.py --pickle-dir ./data --dataset pathfinder32
```

## Example: Modifying ListOps Training Script

```python
# In lra_benchmarks/listops/train.py

# Replace this:
from lra_benchmarks.listops import input_pipeline

# With this:
import os
from pathlib import Path

pickle_dir = Path('./data')
if (pickle_dir / 'lra-listops.train.pickle').exists():
    from lra_benchmarks.listops import input_pipeline_pickle as input_pipeline
    tf.logging.info('Using pickle input pipeline')
else:
    from lra_benchmarks.listops import input_pipeline
    tf.logging.info('Using original input pipeline')

# Rest of the code remains the same
train_dataset, val_dataset, test_dataset, encoder = input_pipeline.get_datasets(
    n_devices=FLAGS.n_devices,
    task_name=FLAGS.task_name,
    data_dir=FLAGS.data_dir,
    batch_size=FLAGS.batch_size,
    max_length=FLAGS.max_length
)
```

## Notes

- The pickle pipelines maintain the same API as the original pipelines
- All preprocessing (tokenization, encoding, padding) is assumed to be done
- Padding token must be 0 (for LRA compatibility)
- Content tokens must be > 0 (for LRA mask compatibility)
- Image data should be flattened to 1D sequences
