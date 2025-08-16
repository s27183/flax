"""CRF: Policy-only PPO variant (no value loss).

Sets vf_coeff=0.0 to disable value loss; suitable for ablation against standard PPO.
"""
import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # Game and training budget
  config.game = 'Pong'
  config.total_frames = 5_000_000
  # Optimizer
  config.learning_rate = 2.5e-4
  # Data shape
  config.batch_size = 256
  config.num_agents = 8
  config.actor_steps = 128
  # Optimization
  config.num_epochs = 3
  config.gamma = 0.99
  config.lambda_ = 0.95
  config.clip_param = 0.1
  # Loss weights
  config.vf_coeff = 0.0  # disable value loss
  config.entropy_coeff = 0.01
  # Schedules
  config.decaying_lr_and_clip_param = True
  return config
