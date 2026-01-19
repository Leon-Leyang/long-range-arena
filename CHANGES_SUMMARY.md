# Summary of Changes

## 1. Pickle Data Support

All training scripts have been updated to use pickle input pipelines instead of the original data loading:

### Files Modified for Pickle Support
- `lra_benchmarks/listops/train.py` - Uses `input_pipeline_pickle`
- `lra_benchmarks/text_classification/train.py` - Uses `input_pipeline_pickle`
- `lra_benchmarks/matching/train.py` - Uses `input_pipeline_pickle`
- `lra_benchmarks/image/task_registry.py` - Uses `input_pipeline_pickle`

### Files Created for Pickle Loading
- `lra_benchmarks/listops/input_pipeline_pickle.py`
- `lra_benchmarks/text_classification/input_pipeline_pickle.py`
- `lra_benchmarks/matching/input_pipeline_pickle.py`
- `lra_benchmarks/image/input_pipeline_pickle.py`

## 2. Modernized for Latest Flax/JAX APIs

The codebase has been completely refactored to use modern APIs (Flax 0.8+, JAX 0.4+, Optax):

### Requirements Updated (`requirements.txt`)
```
jax>=0.4.20
jaxlib>=0.4.20
flax>=0.8.0
optax>=0.1.7
orbax-checkpoint>=0.4.0
ml-collections>=0.1.1
tensorboard>=2.15.0
tensorboardX>=2.6.0
tensorflow>=2.15.0
tensorflow-datasets>=4.9.0
numpy>=1.24.0
absl-py>=2.0.0
```

### Key API Changes

#### 1. Flax Linen API (replacing `flax.deprecated.nn`)

**Before:**
```python
from flax.deprecated import nn

class MyModule(nn.Module):
    def apply(self, x, features):
        return nn.Dense(x, features)
```

**After:**
```python
from flax import linen as nn

class MyModule(nn.Module):
    features: int
    
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)
```

#### 2. Optax (replacing `flax.optim`)

**Before:**
```python
from flax import optim
optimizer_def = optim.Adam(learning_rate)
optimizer = optimizer_def.create(model)
new_optimizer = optimizer.apply_gradient(grad)
```

**After:**
```python
import optax
tx = optax.adamw(learning_rate)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
state = state.apply_gradients(grads=grads)
```

#### 3. TrainState (replacing `nn.Model` + optimizer)

**Before:**
```python
model = nn.Model(module, initial_params)
optimizer = optimizer_def.create(model)
```

**After:**
```python
from flax.training import train_state

class TrainState(train_state.TrainState):
    dropout_rng: Any = None

state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    dropout_rng=dropout_rng
)
```

#### 4. Dropout handling

**Before:**
```python
with nn.stochastic(dropout_rng):
    logits = model(inputs, train=True)
```

**After:**
```python
logits = state.apply_fn(
    {'params': params}, 
    inputs, 
    train=True,
    rngs={'dropout': dropout_rng}
)
```

#### 5. Checkpointing (using Orbax)

**Before:**
```python
from flax.training import checkpoints
checkpoints.save_checkpoint(dir, optimizer, step)
optimizer = checkpoints.restore_checkpoint(dir, optimizer)
```

**After:**
```python
import orbax.checkpoint as ocp
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(path, state)
state = checkpointer.restore(path, state)
```

### Files Completely Rewritten

1. **`lra_benchmarks/models/layers/common_layers.py`**
   - All layer classes converted to Linen API
   - `Embed`, `AddPositionEmbs`, `MlpBlock` classes
   - `ClassifierHead`, `ClassifierHeadDual` as modules

2. **`lra_benchmarks/models/transformer/transformer.py`**
   - `TransformerBlock` - Linen module
   - `TransformerEncoder` - Linen module
   - `TransformerDualEncoder` - Linen module

3. **`lra_benchmarks/utils/train_utils.py`**
   - `get_model_class()` - Returns model class
   - `create_model()` - Creates and initializes model
   - Model registry for available models

4. **All `train.py` files**
   - `lra_benchmarks/listops/train.py`
   - `lra_benchmarks/text_classification/train.py`
   - `lra_benchmarks/matching/train.py`
   - `lra_benchmarks/image/train.py`

## 3. Currently Supported Models

Only the base Transformer model has been updated to the new API. Other models (Performer, Reformer, etc.) will need similar updates before they can be used.

Supported:
- `transformer` - Standard Transformer encoder
- `transformer_dual` - Dual encoder for matching tasks

Not yet updated (will raise error):
- `synthesizer`
- `reformer`
- `performer`
- `linformer`
- `local`
- `bigbird`
- `sinkhorn`
- `linear_transformer`
- `sparse_transformer`
- `longformer`
- `transformer_tlb`

## Reverting Changes

To revert to original data loading (requires old Flax version):

```python
# ListOps
from lra_benchmarks.listops import input_pipeline

# Text Classification  
from lra_benchmarks.text_classification import input_pipeline

# Matching
from lra_benchmarks.matching import input_pipeline

# Image
from lra_benchmarks.image import input_pipeline
```

Note: This will only work with the old `flax.deprecated.nn` API which requires Flax < 0.4.
