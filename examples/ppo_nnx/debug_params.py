#!/usr/bin/env python3
"""Debug script to understand NNX parameter state structure."""

import jax
import jax.numpy as jnp
from flax import nnx
import models

# Create model
key = jax.random.key(42)
rngs = nnx.Rngs(key)
model = models.ActorCritic(num_outputs=6, rngs=rngs)

# Get parameters
params = nnx.state(model, nnx.Param)
print(f"Parameter state type: {type(params)}")
print(f"Parameter state keys: {list(params.keys())}")

# Inspect first parameter
first_key = list(params.keys())[0]
first_param = params[first_key]
print(f"\nFirst parameter '{first_key}':")
print(f"  Type: {type(first_param)}")
print(f"  Dir: {[attr for attr in dir(first_param) if not attr.startswith('_')]}")

# Try different ways to access the actual array
try:
    print(f"  Has 'value': {hasattr(first_param, 'value')}")
    if hasattr(first_param, 'value'):
        print(f"  Value type: {type(first_param.value)}")
        print(f"  Value shape: {first_param.value.shape}")
except Exception as e:
    print(f"  Error accessing value: {e}")

# Try accessing as array directly
try:
    print(f"  Direct array access: {jnp.array(first_param).shape}")
except Exception as e:
    print(f"  Error with direct array access: {e}")

# Explore nested structure
print(f"\nExploring nested structure:")
for layer_name, layer_state in params.items():
    print(f"\nLayer '{layer_name}':")
    print(f"  Type: {type(layer_state)}")
    print(f"  Keys: {list(layer_state.keys())}")

    # Look inside each layer
    for param_name, param_value in layer_state.items():
        print(f"    {param_name}: {type(param_value)}")
        if hasattr(param_value, 'value'):
            print(f"      Value shape: {param_value.value.shape}")
            print(f"      Value size: {param_value.value.size}")
        elif hasattr(param_value, 'shape'):
            print(f"      Shape: {param_value.shape}")
            print(f"      Size: {param_value.size}")

# Try to count parameters correctly
print(f"\nCounting parameters:")
total_params = 0
for layer_name, layer_state in params.items():
    layer_params = 0
    for param_name, param_value in layer_state.items():
        if hasattr(param_value, 'value'):
            param_size = param_value.value.size
        elif hasattr(param_value, 'size'):
            param_size = param_value.size
        else:
            param_size = 0
        layer_params += param_size
        print(f"  {layer_name}.{param_name}: {param_size:,} params")
    total_params += layer_params
    print(f"  {layer_name} total: {layer_params:,} params")

print(f"\nTotal parameters: {total_params:,}")
