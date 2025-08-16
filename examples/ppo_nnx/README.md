# PPO with Flax NNX

This directory contains a complete migration of the PPO (Proximal Policy Optimization) implementation from Flax Linen to the modern Flax NNX API.

## Overview

This implementation demonstrates how to migrate a complex reinforcement learning algorithm from the legacy Flax Linen API to the new Flax NNX API. The migration showcases the benefits of NNX's stateful modules, simplified training loops, and more intuitive object-oriented design.

## Key Differences from Original PPO

### 1. **Stateful Modules**
- **Linen**: Required separate parameter management with `model.init()` and `apply_fn`
- **NNX**: Parameters are embedded in the model, no separate management needed

### 2. **Model Definition**
- **Linen**: Used `@nn.compact` decorator with inline layer definitions
- **NNX**: Uses explicit `__init__` method with layers as attributes

### 3. **Training Loop**
- **Linen**: Complex parameter passing and `train_state.TrainState`
- **NNX**: Simplified with `nnx.Optimizer` and direct model usage

### 4. **JIT Compilation**
- **Linen**: `@functools.partial(jax.jit, static_argnums=...)`
- **NNX**: Clean `@nnx.jit` decorator

## Files Structure

```
ppo_nnx/
├── README.md                           # This file
├── implementation_plan.md              # Detailed migration plan
├── models.py                          # Actor-Critic model (NNX)
├── agent.py                           # Agent utilities (simplified)
├── ppo_lib.py                         # Core training logic (NNX)
├── ppo_main.py                        # Main training script
├── test_episodes.py                   # Policy testing utilities
├── env_utils.py                       # Environment utilities
├── seed_rl_atari_preprocessing.py     # Atari preprocessing
└── configs/
    └── default.py                     # Default hyperparameters
```

## Usage

### Basic Training

```bash
# From the flax root directory
cd examples/ppo_nnx

# Run with default configuration (Pong)
python ppo_main.py --workdir=/tmp/ppo_nnx_training

# Run with different game
python ppo_main.py --config.game=Breakout --workdir=/tmp/ppo_nnx_breakout
```

### Configuration

The training can be configured through the `configs/default.py` file or by overriding specific parameters:

```bash
python ppo_main.py \
  --config.game=Breakout \
  --config.total_frames=10000000 \
  --config.learning_rate=1e-4 \
  --config.num_agents=4 \
  --workdir=/tmp/ppo_nnx_custom
```

## Key Implementation Highlights

### Model Architecture (models.py)

```python
class ActorCritic(nnx.Module):
  def __init__(self, num_outputs: int, *, rngs: nnx.Rngs):
    # Explicit layer definitions as attributes
    self.conv1 = nnx.Conv(in_features=4, out_features=32, ...)
    self.conv2 = nnx.Conv(in_features=32, out_features=64, ...)
    # ... more layers
    
  def __call__(self, x):
    # Direct layer calls, no parameter passing
    x = jax.nn.relu(self.conv1(x))
    x = jax.nn.relu(self.conv2(x))
    # ... forward pass
```

### Simplified Training (ppo_lib.py)

```python
@nnx.jit
def train_step(optimizer: nnx.Optimizer, trajectories, batch_size, **kwargs):
  for batch in zip(*trajectories):
    loss, grads = nnx.value_and_grad(loss_fn)(
        optimizer.model, batch, **kwargs
    )
    optimizer.update(grads)
  return optimizer, total_loss
```

### Direct Model Usage (agent.py)

```python
@nnx.jit
def policy_action(model: ActorCritic, state):
  return model(state)  # No apply_fn or params needed
```

## Benefits of NNX Migration

1. **Simplified Code**: Eliminated complex parameter management
2. **Better Debugging**: Direct access to model weights and state
3. **Modern Patterns**: Object-oriented design with type hints
4. **Performance**: More efficient JIT compilation and memory usage
5. **Maintainability**: Cleaner, more intuitive code structure

## Dependencies

- JAX
- Flax (with NNX support)
- Gymnasium
- ALE-Py (Atari Learning Environment)
- OpenCV (cv2)
- NumPy
- Optax
- TensorFlow (for GPU memory management)
- ML Collections

## Known Limitations

- Checkpoint loading/saving is temporarily disabled (marked with TODOs)
- Random key management in experience collection could be improved
- Full numerical equivalence with original implementation not yet verified

## Future Work

1. Implement checkpoint loading/saving for NNX models
2. Add comprehensive testing to ensure numerical equivalence
3. Performance benchmarking against original Linen implementation
4. Add support for distributed training
5. Implement model inspection and visualization tools

## References

- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/nnx/index.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Original Linen Implementation](../ppo/)

This implementation serves as a comprehensive example of migrating complex ML workloads from Flax Linen to NNX, demonstrating best practices and the benefits of the new API.

---

## RRL: Reversed RL for System Engineers

For a compact, system‑engineering‑first abstraction (shapes → operators → loss), see:

- examples/ppo_nnx/docs/rrl.md

It includes a capability table, operator manifests, pseudocode, and two runnable configs to compare PPO with and without a value network:

```bash
python -m examples.ppo_nnx.ppo_main \
  --config=examples/ppo_nnx/configs/crf_with_value.py \
  --workdir=/tmp/ppo_crf_value

python -m examples.ppo_nnx.ppo_main \
  --config=examples/ppo_nnx/configs/crf_policy_only.py \
  --workdir=/tmp/ppo_crf_policy_only
```
