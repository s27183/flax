# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class and functions to define and initialize the actor-critic model using Flax NNX."""

from flax import nnx
import jax
import jax.numpy as jnp


class ActorCritic(nnx.Module):
  """Class defining the actor-critic model using Flax NNX.

  This is a migration from the Linen-based implementation to the new NNX API.
  The model architecture remains the same but uses the stateful NNX modules.
  """

  def __init__(self, num_outputs: int, *, rngs: nnx.Rngs):
    """Initialize the actor-critic model.

    Args:
      num_outputs: Number of action outputs (depends on the environment)
      rngs: Random number generator state for parameter initialization
    """
    self.num_outputs = num_outputs
    dtype = jnp.float32

    # Define convolutional layers
    # Architecture originates from "Human-level control through deep reinforcement
    # learning.", Nature 518, no. 7540 (2015): 529-533.
    self.conv1 = nnx.Conv(
        in_features=4,  # input channels (stacked frames)
        out_features=32,
        kernel_size=(8, 8),
        strides=(4, 4),
        dtype=dtype,
        rngs=rngs
    )
    self.conv2 = nnx.Conv(
        in_features=32,
        out_features=64,
        kernel_size=(4, 4),
        strides=(2, 2),
        dtype=dtype,
        rngs=rngs
    )
    self.conv3 = nnx.Conv(
        in_features=64,
        out_features=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        dtype=dtype,
        rngs=rngs
    )

    # Define fully connected layers
    # Note: 64 * 11 * 11 is calculated from the conv output dimensions
    # Input: 84x84x4 -> conv1: 21x21x32 -> conv2: 11x11x64 -> conv3: 11x11x64
    self.hidden = nnx.Linear(
        in_features=64 * 11 * 11,
        out_features=512,
        dtype=dtype,
        rngs=rngs
    )

    # Policy head (actor)
    self.logits = nnx.Linear(
        in_features=512,
        out_features=num_outputs,
        dtype=dtype,
        rngs=rngs
    )

    # Value head (critic)
    self.value = nnx.Linear(
        in_features=512,
        out_features=1,
        dtype=dtype,
        rngs=rngs
    )

  def __call__(self, x):
    """Define the forward pass of the actor-critic network.

    Network is used to both estimate policy (logits) and expected state value;
    in other words, hidden layers' params are shared between policy and value
    networks, see e.g.:
    github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py

    Args:
      x: Input tensor of shape (batch_size, 84, 84, 4)

    Returns:
      Tuple of (policy_log_probabilities, value) where:
        - policy_log_probabilities: Log probabilities for each action
        - value: State value estimate
    """
    dtype = jnp.float32
    # Normalize pixel values to [0, 1]
    x = x.astype(dtype) / 255.0

    # Convolutional layers with ReLU activation
    x = jax.nn.relu(self.conv1(x))
    x = jax.nn.relu(self.conv2(x))
    x = jax.nn.relu(self.conv3(x))

    # Flatten for fully connected layers
    x = x.reshape((x.shape[0], -1))

    # Hidden layer with ReLU activation
    x = jax.nn.relu(self.hidden(x))

    # Policy head: compute action logits and convert to log probabilities
    logits = self.logits(x)
    policy_log_probabilities = jax.nn.log_softmax(logits)

    # Value head: estimate state value
    value = self.value(x)

    return policy_log_probabilities, value
