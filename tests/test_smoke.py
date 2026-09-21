"""Small correctness checks for the public PPO learning demo."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Normal


SOURCE_DIR = Path(__file__).resolve().parents[1] / "simplified_demo"
sys.path.insert(0, str(SOURCE_DIR))

from env import ToyResourceEnv
from ppo import ActorCritic, PPOAgent, Rollout
from train import main as train_main


class PPODemoSmokeTests(unittest.TestCase):
    def test_environment_state_shape(self) -> None:
        env = ToyResourceEnv(horizon=4)
        env.seed(7)
        state = env.reset()
        next_state, _, _, info = env.step(direction=1, magnitude=0.5)
        self.assertEqual(state.shape, (3,))
        self.assertEqual(next_state.shape, (3,))
        self.assertEqual(info["effective_action"], 0.5)

    def test_actor_critic_output_and_action_range(self) -> None:
        model = ActorCritic(state_dim=3, hidden_dim=64)
        direction_logits, magnitude_mean, std, value = model(torch.zeros(2, 3))
        self.assertEqual(direction_logits.shape, (2, 2))
        self.assertEqual(magnitude_mean.shape, (2, 1))
        self.assertEqual(std.shape, (2, 1))
        self.assertEqual(value.shape, (2,))
        agent = PPOAgent(3, 64, 3e-4, 1e-3, seed=7)
        direction, magnitude, _, _ = agent.act(np.zeros(3, dtype=np.float32))
        self.assertIn(direction, (-1, 1))
        self.assertGreater(magnitude, 0.0)
        self.assertLess(magnitude, 1.0)

    def test_sigmoid_squashed_log_prob_round_trip_is_consistent(self) -> None:
        """logit(sigmoid(z)) must preserve the continuous log-probability used by PPO."""
        dist = Normal(
            torch.tensor([[0.2], [-0.3]], dtype=torch.float32),
            torch.tensor([[0.7], [0.5]], dtype=torch.float32),
        )
        pre_sigmoid_magnitudes = torch.tensor([[0.4], [-0.6]], dtype=torch.float32)
        magnitudes = torch.sigmoid(pre_sigmoid_magnitudes)
        original_log_probs = PPOAgent._sigmoid_squashed_log_prob(dist, pre_sigmoid_magnitudes)
        recovered_pre_sigmoid_magnitudes = torch.logit(magnitudes)
        recovered_log_probs = PPOAgent._sigmoid_squashed_log_prob(dist, recovered_pre_sigmoid_magnitudes)
        torch.testing.assert_close(original_log_probs, recovered_log_probs, atol=1e-6, rtol=1e-6)

    def test_short_ppo_update_is_finite_and_updates_parameters(self) -> None:
        env = ToyResourceEnv(horizon=8)
        env.seed(7)
        agent = PPOAgent(3, 32, 3e-4, 1e-3, seed=7)
        state = env.reset()
        rollout = Rollout([], [], [], [], [], [], [])
        for _ in range(env.horizon):
            direction, magnitude, log_prob, value = agent.act(state)
            next_state, reward, done, _ = env.step(direction, magnitude)
            rollout.states.append(state)
            rollout.directions.append(direction)
            rollout.magnitudes.append(magnitude)
            rollout.rewards.append(reward)
            rollout.dones.append(done)
            rollout.log_probs.append(log_prob)
            rollout.values.append(value)
            state = next_state
            if done:
                break
        before = [parameter.detach().clone() for parameter in agent.model.parameters()]
        loss = agent.update(rollout, 0.99, 0.95, 0.2, 0.01, epochs=1, batch_size=8)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in agent.model.parameters()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, agent.model.parameters())))

    def test_training_entrypoint_writes_expected_artifacts(self) -> None:
        """A short end-to-end run catches integration errors in rollout collection."""
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "logs"
            with patch.object(sys, "argv", ["train.py", "--episodes", "2", "--seed", "7", "--output-dir", str(output_dir)]):
                train_main()
            self.assertTrue((output_dir / "training.csv").is_file())
            self.assertTrue((output_dir / "evaluation_summary.json").is_file())
            self.assertTrue((output_dir / "training_curve.png").is_file())


if __name__ == "__main__":
    unittest.main()
