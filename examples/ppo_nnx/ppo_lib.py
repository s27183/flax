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

"""PPO training loop and utilities using Flax NNX."""

import functools
from typing import Any

from absl import logging
from flax import nnx
import agent
import models
import test_episodes
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


def gae_advantages(
    rewards: np.ndarray,
    terminal_masks: np.ndarray,
    values: np.ndarray,
    discount: float,
    gae_param: float,
):
  """Use Generalized Advantage Estimation (GAE) to compute advantages.

  As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
  key observation that the advantage function A_{t} can be written as a discounted
  sum of deltas V_{t} - V_{t+1}.

  Args:
    rewards: array shaped (actor_steps,), rewards from the environment
    terminal_masks: array shaped (actor_steps,), zeros for terminal states
    values: array shaped (actor_steps,), values estimated by critic
    discount: RL discount usually denoted with gamma
    gae_param: GAE parameter usually denoted with lambda

  Returns:
    advantages: calculated advantages shaped (actor_steps,)
  """
  assert rewards.shape[0] + 1 == values.shape[0], (
      'One more value needed; Eq. (12) in PPO paper requires '
      f'V(s_{{t+1}}) in \delta_t calculation. rewards.shape={rewards.shape}, '
      f'values.shape={values.shape}'
  )
  advantages = []
  gae = 0.0
  for t in reversed(range(len(rewards))):
    # Masks used to set next state value to 0.0 for terminal states.
    value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
    delta = rewards[t] + value_diff
    # Masks[t] used to ensure that values before and after a terminal state
    # are independent of each other.
    gae = delta + discount * gae_param * terminal_masks[t] * gae
    advantages.append(gae)
  advantages = advantages[::-1]
  return jnp.array(advantages)


def loss_fn(
    model: models.ActorCritic,
    minibatch: tuple,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
):
  """Evaluate the loss function.

  Compute loss as a sum of three components: the negative of the PPO clipped
  surrogate objective, the value function loss and the negative of the entropy
  bonus.

  Args:
    model: the actor-critic model (NNX module)
    minibatch: tuple of five elements forming one experience batch:
               states: shape (batch_size, 84, 84, 4)
               actions: shape (batch_size, 84, 84, 4)
               old_log_probs: shape (batch_size,)
               returns: shape (batch_size,)
               advantages: shape (batch_size,)
    clip_param: the PPO clipping parameter used to clamp ratios in loss function
    vf_coeff: weighs value function loss in total loss
    entropy_coeff: weighs entropy bonus in the total loss

  Returns:
    loss: the PPO loss, scalar quantity
  """
  states, actions, old_log_probs, returns, advantages = minibatch
  log_probs, values = model(states)
  values = values[:, 0]  # Convert shapes: (batch, 1) to (batch, ).
  probs = jnp.exp(log_probs)

  value_loss = jnp.mean(jnp.square(returns - values), axis=0)

  entropy = jnp.sum(-probs * log_probs, axis=1).mean()

  log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
  ratios = jnp.exp(log_probs_act_taken - old_log_probs)
  # Advantage normalization (following the OpenAI baselines).
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  pg_loss = ratios * advantages
  clipped_loss = advantages * jax.lax.clamp(
      1.0 - clip_param, ratios, 1.0 + clip_param
  )
  ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)

  return ppo_loss + vf_coeff * value_loss - entropy_coeff * entropy


@nnx.jit
def train_step(
    optimizer: nnx.Optimizer,
    trajectories: tuple,
    batch_size: int,
    *,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
):
  """Compilable train step using NNX.

  Runs an entire epoch of training (i.e. the loop over minibatches within
  an epoch is included here for performance reasons).

  Args:
    optimizer: the NNX optimizer containing the model
    trajectories: tuple of the following five elements forming the experience:
                  states: shape (steps_per_agent*num_agents, 84, 84, 4)
                  actions: shape (steps_per_agent*num_agents, 84, 84, 4)
                  old_log_probs: shape (steps_per_agent*num_agents, )
                  returns: shape (steps_per_agent*num_agents, )
                  advantages: (steps_per_agent*num_agents, )
    batch_size: the minibatch size, static argument
    clip_param: the PPO clipping parameter used to clamp ratios in loss function
    vf_coeff: weighs value function loss in total loss
    entropy_coeff: weighs entropy bonus in the total loss

  Returns:
    optimizer: updated optimizer after the parameters update
    loss: loss summed over training steps
  """
  iterations = trajectories[0].shape[0] // batch_size
  trajectories = jax.tree_util.tree_map(
      lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), trajectories
  )
  total_loss = 0.0
  for batch in zip(*trajectories):
    loss, grads = nnx.value_and_grad(loss_fn)(
        optimizer.model, batch, clip_param, vf_coeff, entropy_coeff
    )
    total_loss += loss
    optimizer.update(grads)
  return optimizer, total_loss


def get_experience(
    optimizer: nnx.Optimizer,
    simulators: list[agent.RemoteSimulator],
    steps_per_actor: int,
):
  """Collect experience from agents.

  Runs `steps_per_actor` time steps of the game for each of the `simulators`.
  """
  all_experience = []
  # Range up to steps_per_actor + 1 to get one more value needed for GAE.
  for _ in range(steps_per_actor + 1):
    step_experience = []
    for sim in simulators:
      state = sim.conn.recv()
      log_probs, values = agent.policy_action(optimizer.model, state)
      probs = jnp.exp(log_probs)
      action = jax.random.categorical(jax.random.key(0), log_probs[0])
      log_prob_act_taken = log_probs[0, action]
      sim.conn.send(int(action))
      experience = sim.conn.recv()
      step_experience.append(
          (state[0], action, experience[2], values[0, 0], log_prob_act_taken, experience[3])
      )
    all_experience.append(step_experience)
  return all_experience


def process_experience(
    all_experience: list,
    actor_steps: int,
    num_agents: int,
    gamma: float,
    lambda_: float,
):
  """Process experience for training.

  Args:
    all_experience: collected experience from all agents
    actor_steps: number of steps per actor
    num_agents: number of agents
    gamma: discount factor
    lambda_: GAE parameter

  Returns:
    trajectories: tuple of processed experience arrays
  """
  # Prepare data for training.
  all_experience = np.array(all_experience)
  # all_experience is shaped (actor_steps + 1, num_agents, 6) where the last
  # dimension is (state, action, reward, value, log_prob, done).
  states = all_experience[:-1, :, 0]
  actions = all_experience[:-1, :, 1]
  rewards = all_experience[:-1, :, 2]
  values = all_experience[:, :, 3]
  log_probs = all_experience[:-1, :, 4]
  dones = all_experience[:-1, :, 5]

  # Compute advantages using GAE.
  advantages = []
  returns = []
  for i in range(num_agents):
    terminal_masks = 1.0 - dones[:, i].astype(jnp.float32)
    agent_advantages = gae_advantages(
        rewards[:, i], terminal_masks, values[:, i], gamma, lambda_
    )
    agent_returns = agent_advantages + values[:-1, i]
    advantages.append(agent_advantages)
    returns.append(agent_returns)

  advantages = np.array(advantages).T
  returns = np.array(returns).T
  # After preprocessing, we have:
  # states: (actor_steps, num_agents, 84, 84, 4)
  # actions: (actor_steps, num_agents)
  # log_probs: (actor_steps, num_agents)
  # returns: (actor_steps, num_agents)
  # advantages: (actor_steps, num_agents)
  trajectories = (states, actions, log_probs, returns, advantages)
  trajectory_len = num_agents * actor_steps
  trajectories = tuple(
      map(
          lambda x: np.reshape(x, (trajectory_len,) + x.shape[2:]), trajectories
      )
  )
  return trajectories


def get_initial_model(key: jax.Array, num_outputs: int) -> models.ActorCritic:
  """Initialize the actor-critic model using NNX.
  
  Args:
    key: random key for parameter initialization
    num_outputs: number of action outputs
    
  Returns:
    model: initialized ActorCritic model
  """
  rngs = nnx.Rngs(key)
  model = models.ActorCritic(num_outputs=num_outputs, rngs=rngs)
  return model


def create_optimizer(
    model: models.ActorCritic,
    config: ml_collections.ConfigDict,
    train_steps: int,
) -> nnx.Optimizer:
  """Create optimizer for the model.
  
  Args:
    model: the actor-critic model
    config: configuration object
    train_steps: total number of training steps
    
  Returns:
    optimizer: NNX optimizer
  """
  if config.decaying_lr_and_clip_param:
    lr = optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=0.0,
        transition_steps=train_steps,
    )
  else:
    lr = config.learning_rate
  
  tx = optax.adam(lr)
  optimizer = nnx.Optimizer(model, tx)
  return optimizer


def train(
    model: models.ActorCritic, config: ml_collections.ConfigDict, model_dir: str
):
  """Main training loop.

  Args:
    model: the actor-critic model
    config: object holding hyperparameters and the training information
    model_dir: path to dictionary where checkpoints and logging info are stored

  Returns:
    optimizer: the trained optimizer
  """

  game = config.game + 'NoFrameskip-v4'
  simulators = [agent.RemoteSimulator(game) for _ in range(config.num_agents)]
  summary_writer = tensorboard.SummaryWriter(model_dir)
  summary_writer.hparams(dict(config))
  loop_steps = config.total_frames // (config.num_agents * config.actor_steps)
  log_frequency = 40
  checkpoint_frequency = 500
  # train_step does multiple steps per call for better performance
  # compute number of steps per call here to convert between the number of
  # train steps and the inner number of optimizer steps
  iterations_per_step = (
      config.num_agents * config.actor_steps // config.batch_size
  )

  optimizer = create_optimizer(
      model,
      config,
      loop_steps * config.num_epochs * iterations_per_step,
  )
  
  # TODO: Implement checkpoint loading for NNX
  # optimizer = checkpoints.restore_checkpoint(model_dir, optimizer)
  
  start_step = 0  # For now, always start from 0
  logging.info('Start training from step: %s', start_step)

  for step in range(start_step, loop_steps):
    # Bookkeeping and testing.
    if step % log_frequency == 0:
      score = test_episodes.policy_test(1, optimizer.model, game)
      frames = step * config.num_agents * config.actor_steps
      summary_writer.scalar('game_score', score, frames)
      logging.info(
          'Step %s:\nframes seen %s\nscore %s\n\n', step, frames, score
      )

    # Core training code.
    alpha = (
        1.0 - step / loop_steps if config.decaying_lr_and_clip_param else 1.0
    )
    all_experiences = get_experience(optimizer, simulators, config.actor_steps)
    trajectories = process_experience(
        all_experiences,
        config.actor_steps,
        config.num_agents,
        config.gamma,
        config.lambda_,
    )
    clip_param = config.clip_param * alpha
    for _ in range(config.num_epochs):
      permutation = np.random.permutation(
          config.num_agents * config.actor_steps
      )
      trajectories = tuple(x[permutation] for x in trajectories)
      optimizer, _ = train_step(
          optimizer,
          trajectories,
          config.batch_size,
          clip_param=clip_param,
          vf_coeff=config.vf_coeff,
          entropy_coeff=config.entropy_coeff,
      )
    # TODO: Implement checkpoint saving for NNX
    # if (step + 1) % checkpoint_frequency == 0:
    #   checkpoints.save_checkpoint(model_dir, optimizer, step + 1)
  return optimizer