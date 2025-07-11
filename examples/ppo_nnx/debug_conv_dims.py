#!/usr/bin/env python3
"""Debug script to check conv layer output dimensions."""

import jax
import jax.numpy as jnp
from flax import nnx

# Test conv layer dimensions step by step
key = jax.random.key(42)
rngs = nnx.Rngs(key)
dtype = jnp.float32

# Create individual conv layers
conv1 = nnx.Conv(
    in_features=4,
    out_features=32,
    kernel_size=(8, 8),
    strides=(4, 4),
    dtype=dtype,
    rngs=rngs
)

conv2 = nnx.Conv(
    in_features=32,
    out_features=64,
    kernel_size=(4, 4),
    strides=(2, 2),
    dtype=dtype,
    rngs=rngs
)

conv3 = nnx.Conv(
    in_features=64,
    out_features=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    dtype=dtype,
    rngs=rngs
)

# Test with dummy input
dummy_input = jnp.ones((1, 84, 84, 4), dtype=jnp.float32)
print(f"Input shape: {dummy_input.shape}")

# Apply conv layers step by step
x = dummy_input.astype(dtype) / 255.0
print(f"After normalization: {x.shape}")

x = jax.nn.relu(conv1(x))
print(f"After conv1: {x.shape}")

x = jax.nn.relu(conv2(x))
print(f"After conv2: {x.shape}")

x = jax.nn.relu(conv3(x))
print(f"After conv3: {x.shape}")

# Flatten
x_flat = x.reshape((x.shape[0], -1))
print(f"After flatten: {x_flat.shape}")
print(f"Flattened size: {x_flat.shape[1]}")

# Calculate expected size
expected_size = x.shape[1] * x.shape[2] * x.shape[3]
print(f"Expected flattened size: {expected_size}")