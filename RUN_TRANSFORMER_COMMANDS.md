# Commands to Run Transformer Models with Pickle Data

This document provides the commands to run Transformer models on each LRA task using pickle data.

## Prerequisites

1. Ensure pickle data files are in `./data/` directory (or specify `--data_dir`):
   - `lra-listops.{train,dev,test}.pickle`
   - `lra-text.{train,dev,test}.pickle`
   - `lra-retrieval.{train,dev,test}.pickle`
   - `lra-image.{train,dev,test}.pickle`
   - `lra-pathfinder32-curv_contour_length_14.{train,dev,test}.pickle`

2. Set PYTHONPATH to include the long-range-arena directory:
   ```bash
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

## Notes

- All tasks now use pickle data automatically (modified to use `input_pipeline_pickle`)
- The `--data_dir` should point to the directory containing pickle files
- Default `--data_dir` is `./data` if not specified
- Model checkpoints and logs will be saved to `--model_dir`
- To run evaluation only, add `--test_only=True` flag

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
3. All pickle files exist in the specified data directory

If pickle files are not found, the scripts will raise a `FileNotFoundError` with the expected file path.
