"""Configuration for the public PPO learning demo.

The values below belong only to the synthetic teaching environment.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # 固定 seed 讓每次 demo 的結果比較容易重現。
    seed: int = 7
    # episodes 是完整回合數；horizon 是每回合最多幾個 step。
    episodes: int = 40
    horizon: int = 64
    # PPO 更新與 mini-batch 設定。
    update_epochs: int = 4
    batch_size: int = 64
    # gamma 是未來回報的折扣；gae_lambda 用於折衷 advantage estimate 的 bias 與 variance。
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # PPO clipping 的安全範圍。
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    # Actor 與 Critic 可以使用不同 learning rate。
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    # toy neural network 的 hidden layer 寬度。
    hidden_dim: int = 64
    target: float = 0.8
    step_size: float = 0.15
    evaluation_episodes: int = 10
    output_dir: str = "logs"
