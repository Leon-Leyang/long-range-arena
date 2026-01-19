# Commands to Run Transformer Models with Pickle Data

This document provides the commands to run Transformer models on each LRA task using pickle data.

## Prerequisites

1. Install updated dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure pickle data files are in `./data/` directory:
   - `lra-listops.{train,dev,test}.pickle`
   - `lra-text.{train,dev,test}.pickle`
   - `lra-retrieval.{train,dev,test}.pickle`
   - `lra-image.{train,dev,test}.pickle`
   - `lra-pathfinder32-curv_contour_length_14.{train,dev,test}.pickle`

3. Set PYTHONPATH to include the long-range-arena directory:
   ```bash
   cd /path/to/long-range-arena
   export PYTHONPATH="$(pwd):$PYTHONPATH"
   ```

## Commands

### 1. ListOps Task

```bash
PYTHONPATH="$(pwd):$PYTHONPATH" python lra_benchmarks/listops/train.py \
    --config=lra_benchmarks/listops/configs/transformer_base.py \
    --model_dir=/tmp/listops_transformer \
    --task_name=basic \
    --data_dir=./data
```

### 2. Text Classification (IMDB) Task

```bash
PYTHONPATH="$(pwd):$PYTHONPATH" python lra_benchmarks/text_classification/train.py \
    --config=lra_benchmarks/text_classification/configs/transformer_base.py \
    --model_dir=/tmp/text_transformer \
    --task_name=imdb_reviews \
    --data_dir=./data
```

### 3. Matching/Retrieval (AAN) Task

```bash
PYTHONPATH="$(pwd):$PYTHONPATH" python lra_benchmarks/matching/train.py \
    --config=lra_benchmarks/matching/configs/transformer_base.py \
    --model_dir=/tmp/matching_transformer \
    --task_name=aan \
    --data_dir=./data
```

### 4. CIFAR-10 Image Task

**Note:** Image tasks don't use `--data_dir` flag. Pickle files are loaded from `./data/` by default.

```bash
PYTHONPATH="$(pwd):$PYTHONPATH" python lra_benchmarks/image/train.py \
    --config=lra_benchmarks/image/configs/cifar10/transformer_base.py \
    --model_dir=/tmp/cifar10_transformer \
    --task_name=cifar10
```

### 5. Pathfinder32 Task

```bash
PYTHONPATH="$(pwd):$PYTHONPATH" python lra_benchmarks/image/train.py \
    --config=lra_benchmarks/image/configs/pathfinder32/transformer_base.py \
    --model_dir=/tmp/pathfinder32_transformer \
    --task_name=pathfinder32_easy
```

For intermediate difficulty:
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" python lra_benchmarks/image/train.py \
    --config=lra_benchmarks/image/configs/pathfinder32/transformer_base.py \
    --model_dir=/tmp/pathfinder32_inter_transformer \
    --task_name=pathfinder32_inter
```

For hard difficulty:
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" python lra_benchmarks/image/train.py \
    --config=lra_benchmarks/image/configs/pathfinder32/transformer_base.py \
    --model_dir=/tmp/pathfinder32_hard_transformer \
    --task_name=pathfinder32_hard
```

### 6. Pathfinder128 (PathX) Task

```bash
PYTHONPATH="$(pwd):$PYTHONPATH" python lra_benchmarks/image/train.py \
    --config=lra_benchmarks/image/configs/pathfinder128/transformer_base.py \
    --model_dir=/tmp/pathfinder128_transformer \
    --task_name=pathfinder128_easy
```

## Code Updates Summary

The codebase has been updated to use modern JAX/Flax APIs:

### Key Changes

1. **Flax Linen API**: Replaced `flax.deprecated.nn` with `flax.linen`
   - `nn.Module` with `apply()` → `nn.Module` with `@nn.compact` and `__call__()`
   - `nn.Embed.partial()` → `nn.Embed` as a class attribute
   - `nn.Dense(x, features)` → `nn.Dense(features)(x)`

2. **Optax**: Replaced `flax.optim` with `optax`
   - `optim.Adam()` → `optax.adamw()`
   - `optimizer.apply_gradient()` → `state.apply_gradients()`

3. **TrainState**: Using `flax.training.train_state.TrainState`
   - No more `nn.Model` wrapper
   - Cleaner state management

4. **Checkpointing**: Using `orbax-checkpoint`
   - `checkpoints.save_checkpoint()` → `ocp.StandardCheckpointer().save()`
   - `checkpoints.restore_checkpoint()` → `ocp.StandardCheckpointer().restore()`

5. **TensorBoard**: Using `tensorboardX.SummaryWriter`
   - `flax.metrics.tensorboard` → `tensorboardX`

6. **Dropout**: Passing dropout RNG through `rngs={'dropout': rng}`
   - No more `nn.stochastic()` context manager

## Notes

- All tasks now use pickle data automatically (modified to use `input_pipeline_pickle`)
- The `--data_dir` should point to the directory containing pickle files
- Default `--data_dir` is `./data` if not specified
- Model checkpoints and logs will be saved to `--model_dir`
- To run evaluation only, add `--test_only=True` flag (listops, text, matching) or `--eval_only=True` (image)

## Verification

Before running, verify your pickle files exist:

```bash
ls -lh ./data/lra-*.pickle
```

You should see files like:
- `lra-listops.train.pickle`
- `lra-listops.dev.pickle`
- `lra-listops.test.pickle`
- `lra-text.train.pickle`
- `lra-text.dev.pickle`
- `lra-text.test.pickle`
- `lra-retrieval.train.pickle`
- `lra-retrieval.dev.pickle`
- `lra-retrieval.test.pickle`
- `lra-image.train.pickle`
- `lra-image.dev.pickle`
- `lra-image.test.pickle`
- `lra-pathfinder32-curv_contour_length_14.train.pickle`
- `lra-pathfinder32-curv_contour_length_14.dev.pickle`
- `lra-pathfinder32-curv_contour_length_14.test.pickle`

## Troubleshooting

If you get import errors, make sure:
1. You're running from the `long-range-arena` directory
2. PYTHONPATH includes the current directory
3. All dependencies are installed: `pip install -r requirements.txt`
4. All pickle files exist in the specified data directory

If pickle files are not found, the scripts will raise a `FileNotFoundError` with the expected file path.

### Common Issues

1. **ModuleNotFoundError: No module named 'tensorboardX'**
   ```bash
   pip install tensorboardX
   ```

2. **ModuleNotFoundError: No module named 'optax'**
   ```bash
   pip install optax
   ```

3. **ModuleNotFoundError: No module named 'orbax'**
   ```bash
   pip install orbax-checkpoint
   ```
