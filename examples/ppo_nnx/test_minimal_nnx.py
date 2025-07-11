#!/usr/bin/env python3
"""Minimal NNX functionality test for PPO model (JAX/Flax only)."""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    print("Testing minimal NNX functionality...")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    # Test model import and initialization
    print("\nTesting model import...")
    import models
    print("✓ Models import successful")

    # Test model initialization
    print("\nTesting model initialization...")
    key = jax.random.key(42)
    rngs = nnx.Rngs(key)
    num_actions = 6  # Typical for Atari games

    model = models.ActorCritic(num_outputs=num_actions, rngs=rngs)
    print(f"✓ Model initialized: {type(model)}")
    print(f"  - Model has {num_actions} output actions")

    # Inspect model structure
    print("\nInspecting model structure...")
    print(f"  - conv1: {type(model.conv1)}")
    print(f"  - conv2: {type(model.conv2)}")
    print(f"  - conv3: {type(model.conv3)}")
    print(f"  - hidden: {type(model.hidden)}")
    print(f"  - logits: {type(model.logits)}")
    print(f"  - value: {type(model.value)}")

    # Test model forward pass
    print("\nTesting model forward pass...")
    batch_size = 2
    dummy_input = jnp.ones((batch_size, 84, 84, 4), dtype=jnp.float32)

    log_probs, values = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Log probs shape: {log_probs.shape} (expected: ({batch_size}, {num_actions}))")
    print(f"  - Values shape: {values.shape} (expected: ({batch_size}, 1))")
    print(f"  - Log probs range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
    print(f"  - Values range: [{values.min():.3f}, {values.max():.3f}]")

    # Verify log probabilities are valid
    probs = jnp.exp(log_probs)
    prob_sums = jnp.sum(probs, axis=1)
    print(f"  - Probability sums: {prob_sums} (should be close to 1.0)")

    # Test that probabilities are normalized
    assert jnp.allclose(prob_sums, 1.0, atol=1e-6), f"Probabilities not normalized: {prob_sums}"
    print("  ✓ Probabilities are properly normalized")

    # Test JIT compilation
    print("\nTesting JIT compilation...")

    @nnx.jit
    def jitted_forward(model, x):
        return model(x)

    jitted_log_probs, jitted_values = jitted_forward(model, dummy_input)
    print(f"✓ JIT compilation successful")

    # Verify JIT results match non-JIT results
    assert jnp.allclose(log_probs, jitted_log_probs), "JIT results don't match non-JIT"
    assert jnp.allclose(values, jitted_values), "JIT results don't match non-JIT"
    print("  ✓ JIT results match non-JIT results")

    # Test parameter access
    print("\nTesting parameter access...")
    params = nnx.state(model, nnx.Param)
    print(f"✓ Parameter extraction successful")
    print(f"  - Number of parameter groups: {len(params)}")

    # Count total parameters
    total_params = 0
    for layer_name, layer_state in params.items():
        layer_params = 0
        for param_name, param_value in layer_state.items():
            param_count = param_value.value.size
            layer_params += param_count
            print(f"  - {layer_name}.{param_name}: {param_value.value.shape} ({param_count:,} params)")
        total_params += layer_params

    print(f"  - Total parameters: {total_params:,}")

    # Test different input sizes
    print("\nTesting different batch sizes...")
    for test_batch_size in [1, 4, 8]:
        test_input = jnp.ones((test_batch_size, 84, 84, 4), dtype=jnp.float32)
        test_log_probs, test_values = model(test_input)
        expected_log_shape = (test_batch_size, num_actions)
        expected_val_shape = (test_batch_size, 1)

        assert test_log_probs.shape == expected_log_shape, f"Wrong log_probs shape for batch {test_batch_size}"
        assert test_values.shape == expected_val_shape, f"Wrong values shape for batch {test_batch_size}"

        print(f"  ✓ Batch size {test_batch_size}: log_probs {test_log_probs.shape}, values {test_values.shape}")

    # Test model state consistency
    print("\nTesting model state consistency...")
    # Run forward pass twice to ensure consistent results
    result1 = model(dummy_input)
    result2 = model(dummy_input)

    assert jnp.allclose(result1[0], result2[0]), "Model outputs are not consistent"
    assert jnp.allclose(result1[1], result2[1]), "Model outputs are not consistent"
    print("✓ Model outputs are consistent across calls")

    print("\n" + "="*60)
    print("🎉 ALL MINIMAL NNX FUNCTIONALITY TESTS PASSED!")
    print("="*60)
    print("\nThe PPO NNX model implementation is working correctly.")
    print("Key components verified:")
    print("- ✓ Model initialization with NNX")
    print("- ✓ Forward pass with correct output shapes")
    print("- ✓ Probability distribution normalization")
    print("- ✓ JIT compilation with @nnx.jit")
    print("- ✓ Parameter access and counting")
    print("- ✓ Variable batch size handling")
    print("- ✓ Model state consistency")
    print(f"\nModel summary:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Input shape: (batch, 84, 84, 4)")
    print(f"- Output shapes: (batch, {num_actions}), (batch, 1)")
    print(f"- Architecture: CNN + FC layers for actor-critic")

except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
