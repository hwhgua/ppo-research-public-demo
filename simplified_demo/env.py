"""A deliberately small hybrid-action environment for the PPO learning demo."""

from __future__ import annotations

import numpy as np


class ToyResourceEnv:
    """A small hybrid discrete-continuous control environment.

    The agent first chooses a discrete movement direction (-1 or +1), then a
    continuous movement magnitude in [0, 1]. Their product controls a scalar
    resource level. The environment demonstrates PPO workflow rather than a
    specific real-world system.
    """

    def __init__(self, target: float = 0.8, horizon: int = 64, step_size: float = 0.15):
        self.target = float(target)
        self.horizon = int(horizon)
        self.step_size = float(step_size)
        self.rng = np.random.default_rng()
        self.position = 0.0
        self.step_count = 0

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        # 每個 episode 從不同的初始位置開始，避免 agent 只記住單一答案。
        self.position = float(self.rng.uniform(-1.0, 1.0))
        self.step_count = 0
        return self._state()

    def step(self, direction: int, magnitude: float) -> tuple[np.ndarray, float, bool, dict]:
        # direction 是離散選擇：-1 代表往負方向，+1 代表往正方向。
        # magnitude 是介於 0 與 1 的連續值；兩者相乘後才是實際控制量。
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        magnitude = float(magnitude)
        if not 0.0 <= magnitude <= 1.0:
            raise ValueError("magnitude must be between 0 and 1")
        effective_action = float(direction * magnitude)
        # step_size=0.15、direction=+1、magnitude=0.5 時，主動移動量為 +0.075。
        self.position = float(np.clip(self.position + self.step_size * effective_action, -1.5, 1.5))
        self.step_count += 1
        # 距離目標越遠，平方誤差越大；magnitude cost 鼓勵較平穩的控制。
        distance = self.position - self.target
        reward = -distance * distance - 0.01 * magnitude * magnitude
        done = self.step_count >= self.horizon
        return self._state(), float(reward), done, {"distance": abs(distance), "effective_action": effective_action}

    def _state(self) -> np.ndarray:
        # State 是 Agent 唯一能看到的資訊：目前位置、目標與剩餘時間比例。
        remaining = 1.0 - self.step_count / self.horizon
        return np.array([self.position, self.target, remaining], dtype=np.float32)
