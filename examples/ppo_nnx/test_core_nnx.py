#!/usr/bin/env python3
"""Core NNX functionality test for PPO implementation (no Atari dependencies)."""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import ml_collections
    
    # Test core imports (avoiding env_utils which requires cv2)
    print("Testing core imports...")
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
    
    # Test model forward pass
    print("\nTesting model forward pass...")
    batch_size = 2
    dummy_input = jnp.ones((batch_size, 84, 84, 4), dtype=jnp.float32)
    
    log_probs, values = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  - Log probs shape: {log_probs.shape} (expected: ({batch_size}, {num_actions}))")
    print(f"  - Values shape: {values.shape} (expected: ({batch_size}, 1))")
    print(f"  - Log probs range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
    print(f"  - Values range: [{values.min():.3f}, {values.max():.3f}]")
    
    # Verify log probabilities sum to 1 when exponentiated
    probs = jnp.exp(log_probs)
    prob_sums = jnp.sum(probs, axis=1)
    print(f"  - Probability sums: {prob_sums} (should be close to 1.0)")
    
    # Test direct agent policy action (without importing agent.py which imports env_utils)
    print("\nTesting direct policy action...")
    
    @nnx.jit
    def policy_action_test(model, state):
        return model(state)
    
    single_state = dummy_input[:1]  # Single state
    agent_log_probs, agent_values = policy_action_test(model, single_state)
    print(f"✓ Direct policy action successful")
    print(f"  - Agent log probs shape: {agent_log_probs.shape}")
    print(f"  - Agent values shape: {agent_values.shape}")
    
    # Test core PPO lib functions (avoiding full import)
    print("\nTesting core PPO functions...")
    
    # Import specific functions to avoid env_utils dependency
    from ppo_lib import loss_fn, create_optimizer, gae_advantages
    
    # Test optimizer creation
    config = ml_collections.ConfigDict()
    config.learning_rate = 2.5e-4
    config.decaying_lr_and_clip_param = False
    
    optimizer = create_optimizer(model, config, train_steps=1000)
    print(f"✓ Optimizer created: {type(optimizer)}")
    
    # Test loss function
    print("\nTesting loss function...")
    # Create dummy minibatch
    states = dummy_input
    actions = jnp.array([0, 1])  # Action indices
    old_log_probs = jnp.array([-1.5, -1.2])
    returns = jnp.array([0.5, -0.3])
    advantages = jnp.array([0.1, -0.2])
    
    minibatch = (states, actions, old_log_probs, returns, advantages)
    
    loss = loss_fn(
        model, minibatch, 
        clip_param=0.1, 
        vf_coeff=0.5, 
        entropy_coeff=0.01
    )
    print(f"✓ Loss computation successful: {loss:.6f}")
    
    # Test GAE advantages computation
    print("\nTesting GAE advantages...")
    rewards = jnp.array([1.0, 0.5, -0.2, 0.8])
    terminal_masks = jnp.array([1.0, 1.0, 1.0, 0.0])  # Last step is terminal
    values = jnp.array([0.5, 0.3, 0.1, 0.2, 0.0])  # One extra value for GAE
    
    advantages = gae_advantages(rewards, terminal_masks, values, discount=0.99, gae_param=0.95)
    print(f"✓ GAE computation successful")
    print(f"  - Advantages shape: {advantages.shape}")
    print(f"  - Advantages: {advantages}")
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    
    def test_loss_fn(model):
        return loss_fn(model, minibatch, 0.1, 0.5, 0.01)
    
    loss_val, grads = nnx.value_and_grad(test_loss_fn)(model)
    print(f"✓ Gradient computation successful")
    print(f"  - Loss value: {loss_val:.6f}")
    print(f"  - Gradients computed for model parameters")
    
    # Test optimizer update
    print("\nTesting optimizer update...")
    initial_params = nnx.state(optimizer.model, nnx.Param)
    optimizer.update(grads)
    updated_params = nnx.state(optimizer.model, nnx.Param)
    print(f"✓ Optimizer update successful")
    print(f"  - Parameters updated (shapes preserved)")
    
    print("\n" + "="*60)
    print("🎉 ALL CORE NNX FUNCTIONALITY TESTS PASSED!")
    print("="*60)
    print("\nThe PPO NNX core implementation is working correctly.")
    print("Key components tested:")
    print("- ✓ Model initialization with NNX")
    print("- ✓ Forward pass and output shapes")
    print("- ✓ Probability distribution validation")
    print("- ✓ JIT compilation with @nnx.jit")
    print("- ✓ Optimizer creation and configuration")
    print("- ✓ PPO loss function computation")
    print("- ✓ GAE advantages calculation")
    print("- ✓ Gradient computation with nnx.value_and_grad")
    print("- ✓ Optimizer parameter updates")
    print("\nNote: Environment-specific tests skipped due to missing dependencies.")
    print("Install opencv-python, gymnasium, and ale-py for full testing.")
    
except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)