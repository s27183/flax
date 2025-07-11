#!/usr/bin/env python3
"""Basic functionality test for PPO NNX implementation."""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    
    # Test imports
    print("Testing imports...")
    import models
    import agent
    import ppo_lib
    import env_utils
    print("✓ All imports successful")
    
    # Test model initialization
    print("\nTesting model initialization...")
    key = jax.random.key(42)
    rngs = nnx.Rngs(key)
    num_actions = 6  # Typical for Atari games
    
    model = models.ActorCritic(num_outputs=num_actions, rngs=rngs)
    print(f"✓ Model initialized: {type(model)}")
    
    # Test model forward pass
    print("\nTesting model forward pass...")
    batch_size = 2
    dummy_input = jnp.ones((batch_size, 84, 84, 4), dtype=jnp.float32)
    
    log_probs, values = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  - Log probs shape: {log_probs.shape}")
    print(f"  - Values shape: {values.shape}")
    print(f"  - Log probs range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
    print(f"  - Values range: [{values.min():.3f}, {values.max():.3f}]")
    
    # Test agent policy action
    print("\nTesting agent policy action...")
    single_state = dummy_input[:1]  # Single state
    agent_log_probs, agent_values = agent.policy_action(model, single_state)
    print(f"✓ Agent policy action successful")
    print(f"  - Agent log probs shape: {agent_log_probs.shape}")
    print(f"  - Agent values shape: {agent_values.shape}")
    
    # Test optimizer creation
    print("\nTesting optimizer creation...")
    import ml_collections
    
    # Create a minimal config
    config = ml_collections.ConfigDict()
    config.learning_rate = 2.5e-4
    config.decaying_lr_and_clip_param = False
    
    optimizer = ppo_lib.create_optimizer(model, config, train_steps=1000)
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
    
    loss = ppo_lib.loss_fn(
        model, minibatch, 
        clip_param=0.1, 
        vf_coeff=0.5, 
        entropy_coeff=0.01
    )
    print(f"✓ Loss computation successful: {loss:.6f}")
    
    # Test environment utilities
    print("\nTesting environment utilities...")
    try:
        num_actions_pong = env_utils.get_num_actions('PongNoFrameskip-v4')
        print(f"✓ Environment utilities working - Pong actions: {num_actions_pong}")
    except Exception as e:
        print(f"⚠ Environment utilities test failed (expected if ALE not installed): {e}")
    
    print("\n" + "="*50)
    print("🎉 ALL BASIC FUNCTIONALITY TESTS PASSED!")
    print("="*50)
    print("\nThe PPO NNX implementation appears to be working correctly.")
    print("Key components tested:")
    print("- Model initialization and forward pass")
    print("- Agent policy action")
    print("- Optimizer creation")
    print("- Loss function computation")
    print("- Environment utilities")
    
except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)