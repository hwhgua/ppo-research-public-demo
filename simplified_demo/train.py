"""Readable PPO training entry point for the public demo."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from env import ToyResourceEnv
from ppo import PPOAgent, Rollout


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train PPO on a synthetic public-safe environment")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="logs")
    args = parser.parse_args()
    return Config(episodes=args.episodes, seed=args.seed, output_dir=args.output_dir)


def evaluate(agent: PPOAgent, env: ToyResourceEnv, episodes: int = 10) -> float:
    """Evaluate the most likely direction and mean magnitude without exploration."""
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for _ in range(env.horizon):
            direction, magnitude, _, _ = agent.act(state, deterministic=True)
            state, reward, done, _ = env.step(direction, magnitude)
            total += reward
            if done:
                break
        rewards.append(total)
    return float(np.mean(rewards))


def evaluate_random(env: ToyResourceEnv, episodes: int, seed: int) -> float:
    """Evaluate a uniformly random direction and magnitude baseline."""
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for _ in range(env.horizon):
            direction = -1 if int(rng.integers(0, 2)) == 0 else 1
            magnitude = float(rng.uniform(0.0, 1.0))
            state, reward, done, _ = env.step(direction, magnitude)
            total += reward
            if done:
                break
        rewards.append(total)
    return float(np.mean(rewards))


def main() -> None:
    cfg = parse_args()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = ToyResourceEnv(cfg.target, cfg.horizon, cfg.step_size)
    env.seed(cfg.seed)
    agent = PPOAgent(3, cfg.hidden_dim, cfg.actor_lr, cfg.critic_lr, cfg.seed)
    rows: list[dict[str, float]] = []

    for episode in range(1, cfg.episodes + 1):
        # 一個 episode：reset → 收集 rollout → PPO update。
        state = env.reset()
        rollout = Rollout([], [], [], [], [], [], [])
        total_reward = 0.0
        for _ in range(cfg.horizon):
            # Actor 選擇離散 direction 與連續 magnitude；Environment 回傳下一個 state 與 reward。
            direction, magnitude, log_prob, value = agent.act(state)
            next_state, reward, done, _ = env.step(direction, magnitude)
            rollout.states.append(state)
            rollout.directions.append(direction)
            rollout.magnitudes.append(magnitude)
            rollout.rewards.append(reward)
            rollout.dones.append(done)
            rollout.log_probs.append(log_prob)
            rollout.values.append(value)
            total_reward += reward
            state = next_state
            if done:
                break
        # 收集完一回合後，才用這批經驗更新 Actor 與 Critic。
        loss = agent.update(
            rollout,
            cfg.gamma,
            cfg.gae_lambda,
            cfg.clip_epsilon,
            cfg.entropy_coef,
            cfg.update_epochs,
            cfg.batch_size,
        )
        rows.append({"episode": episode, "reward": total_reward, "loss": loss})

    eval_env = ToyResourceEnv(cfg.target, cfg.horizon, cfg.step_size)
    eval_env.seed(cfg.seed + 1)
    eval_reward = evaluate(agent, eval_env, cfg.evaluation_episodes)
    random_env = ToyResourceEnv(cfg.target, cfg.horizon, cfg.step_size)
    random_env.seed(cfg.seed + 1)
    random_reward = evaluate_random(random_env, cfg.evaluation_episodes, cfg.seed + 2)

    csv_path = out_dir / "training.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "reward", "loss"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "episodes": cfg.episodes,
        "seed": cfg.seed,
        "evaluation_episodes": cfg.evaluation_episodes,
        "ppo_deterministic_mean_reward": eval_reward,
        "random_policy_mean_reward": random_reward,
    }
    with (out_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot([row["episode"] for row in rows], [row["reward"] for row in rows], label="episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.axhline(random_reward, color="tab:red", linestyle="--", label=f"random baseline ({random_reward:.2f})")
    plt.axhline(eval_reward, color="tab:green", linestyle="--", label=f"PPO evaluation ({eval_reward:.2f})")
    plt.title("Synthetic hybrid-action PPO training")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=160)
    print(f"Wrote {csv_path}")
    print(f"Deterministic PPO evaluation mean reward: {eval_reward:.3f}")
    print(f"Random policy mean reward: {random_reward:.3f}")


if __name__ == "__main__":
    main()
