"""Recurrent visual PPO, with CUDA rollout storage and explicit timeout handling.

Environment contract: reset() returns image/proprio tensors; step(tanh(action))
returns observations, reward, done, info, and resets done environments. For a
time limit, info['timeouts'] marks truncations and info['terminal_observation']
contains the observations *before* reset. No diagnostics enter either network.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .policy import RecurrentActorCritic


@dataclass
class PPOConfig:
    total_timesteps: int = 1_000_000
    rollout_steps: int = 64
    sequence_length: int = 16
    minibatch_sequences: int = 16
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden_size: int = 128
    seed: int = 0
    torch_threads: int = 4
    checkpoint_every_updates: int = 20
    checkpoint_every_seconds: float = 60.0
    callback_every_steps: int = 25
    max_walltime_seconds: float | None = None

    def validate(self) -> None:
        for name in (
            "total_timesteps", "rollout_steps", "sequence_length", "minibatch_sequences",
            "update_epochs", "hidden_size", "torch_threads", "checkpoint_every_updates",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rollout_steps % self.sequence_length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("invalid gamma or gae_lambda")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.clip_coef <= 0:
            raise ValueError("learning_rate, max_grad_norm and clip_coef must be positive")
        if self.max_walltime_seconds is not None and self.max_walltime_seconds <= 0:
            raise ValueError("max_walltime_seconds must be positive")


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """GAE after timeout terminal values have been added to truncation rewards."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(last_value)
    for step in reversed(range(rewards.shape[0])):
        nonterminal = (~dones[step]).to(rewards.dtype)
        next_value = last_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * nonterminal * next_value - values[step]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[step] = last_advantage
    return advantages, advantages + values


def _chunks(tensor: Tensor, length: int) -> Tensor:
    """[T, N, ...] -> [sequence_length, sequences, ...], without mixing envs."""
    steps, envs = tensor.shape[:2]
    shaped = tensor.reshape(steps // length, length, envs, *tensor.shape[2:])
    return shaped.transpose(0, 1).reshape(length, -1, *tensor.shape[2:])


def train(
    env: Any,
    run_dir: str | Path,
    config: PPOConfig | None = None,
    callback: Callable[[dict, RecurrentActorCritic, Any], None] | None = None,
    resume: str | Path | None = None,
    *,
    init_checkpoint: str | Path | None = None,
) -> Path:
    """Train and return the final checkpoint path.

    callback(metrics, policy, env) receives startup, rollout_step, update,
    checkpoint and complete events, allowing the caller to update a real desktop
    viewer without coupling rendering to learning. Resume restores the model,
    optimizer, counters and RNGs; simulation and recurrent states restart at a
    reset boundary, since an environment snapshot is not part of this contract.
    """
    if resume is not None and init_checkpoint is not None:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    config = config or PPOConfig()
    config.validate()
    torch.set_num_threads(config.torch_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(env.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ppo_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    obs = env.reset()
    num_envs = int(env.num_envs)
    image_shape = tuple(obs["image"].shape[1:])
    proprio_dim = int(obs["proprio"].shape[-1])
    policy = RecurrentActorCritic(
        image_shape=image_shape, proprio_dim=proprio_dim, hidden_size=config.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
    global_step = 0
    first_update = 0
    counters: Counter = Counter()
    recent_returns: deque = deque(maxlen=100)
    recent_lengths: deque = deque(maxlen=100)
    if resume is not None or init_checkpoint is not None:
        checkpoint = torch.load(resume or init_checkpoint, map_location=device, weights_only=False)
        if checkpoint["network_kwargs"] != policy.network_kwargs():
            raise ValueError("checkpoint architecture does not match the environment/config")
        policy.load_state_dict(checkpoint["model"])
    if resume is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        _restore_rng(checkpoint["rng_state"])
        global_step = int(checkpoint["global_step"])
        first_update = int(checkpoint["update"])
        counters.update(checkpoint.get("event_counts", {}))
        recent_returns.extend(checkpoint.get("recent_returns", []))
        recent_lengths.extend(checkpoint.get("recent_lengths", []))
    hidden = policy.initial_hidden(num_envs, device)
    episode_start = torch.ones(num_envs, dtype=torch.bool, device=device)
    episode_returns = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    steps = config.rollout_steps
    shape = (steps, num_envs)
    images = torch.empty((*shape, *image_shape), dtype=torch.float16, device=device)
    proprio = torch.empty((*shape, proprio_dim), device=device)
    hidden_states = torch.empty((*shape, config.hidden_size), device=device)
    starts = torch.empty(shape, dtype=torch.bool, device=device)
    raw_actions = torch.empty((*shape, 19), device=device)
    log_probs = torch.empty(shape, device=device)
    rewards = torch.empty(shape, device=device)
    values = torch.empty(shape, device=device)
    dones = torch.empty(shape, dtype=torch.bool, device=device)
    start_time = time.perf_counter()
    initial_step = global_step
    last_checkpoint_time = start_time
    last_metrics: dict = {}
    current_update = first_update

    def notify(event: str, metrics: dict) -> None:
        if callback is not None:
            callback({**metrics, "event": event, "global_step": global_step}, policy, env)

    def save_checkpoint(name: str) -> Path:
        nonlocal last_checkpoint_time
        path = run_dir / name
        temporary = path.with_suffix(".tmp")
        torch.save({
            "format_version": 1,
            "network_kwargs": policy.network_kwargs(),
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "environment": env.config_dict() if hasattr(env, "config_dict") else None,
            "initialization_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "global_step": global_step,
            "update": current_update,
            "rng_state": _rng_state(),
            "event_counts": dict(counters),
            "recent_returns": list(recent_returns),
            "recent_lengths": list(recent_lengths),
            "metrics": last_metrics,
            "resume_semantics": "environment and GRU reset; optimizer and random states restored",
        }, temporary)
        os.replace(temporary, path)
        last_checkpoint_time = time.perf_counter()
        notify("checkpoint", {**last_metrics, "checkpoint": str(path)})
        return path

    notify("startup", {
        "device": str(device), "gpu_training": device.type == "cuda",
        "num_envs": num_envs, "image_shape": image_shape,
        "resume": str(resume) if resume else None,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
    })
    final_path = run_dir / "checkpoint_final.pt"
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as metrics_file:
        try:
            while global_step < config.total_timesteps:
                current_update += 1
                rollout_start = time.perf_counter()
                raw_reward_sum = torch.zeros((), device=device)
                update_events: Counter = Counter()
                episode_diagnostics: dict[str, list[float]] = {}
                policy.eval()
                for step in range(steps):
                    images[step].copy_(obs["image"])
                    proprio[step].copy_(obs["proprio"])
                    hidden_states[step].copy_(hidden)
                    starts[step].copy_(episode_start)
                    with torch.no_grad():
                        action, log_prob, value, next_hidden = policy.act(obs, hidden, episode_start)
                        next_obs, reward, done, info = env.step(action.tanh())
                        reward = reward.to(device=device, dtype=torch.float32)
                        done = done.to(device=device, dtype=torch.bool)
                        if not torch.isfinite(reward).all():
                            raise FloatingPointError("environment returned a non-finite reward")
                        timeout = info.get("timeouts", torch.zeros_like(done)).to(device).bool()
                        if (timeout & ~done).any():
                            raise ValueError("timeouts must be a subset of done")
                        adjusted_reward = reward.clone()
                        if timeout.any():
                            terminal_obs = info.get("terminal_observation")
                            if terminal_obs is None:
                                raise ValueError("timeouts require terminal_observation before auto-reset")
                            terminal_obs = {
                                key: terminal_obs[key][timeout] for key in ("image", "proprio")
                            }
                            terminal_value = policy.value(
                                terminal_obs, next_hidden[timeout], torch.zeros_like(done[timeout])
                            )
                            adjusted_reward[timeout] += config.gamma * terminal_value
                        raw_actions[step].copy_(action)
                        log_probs[step].copy_(log_prob)
                        values[step].copy_(value)
                        rewards[step].copy_(adjusted_reward)
                        dones[step].copy_(done)
                        raw_reward_sum += reward.sum()
                        episode_returns += reward
                        episode_lengths += 1
                        if done.any():
                            recent_returns.extend(episode_returns[done].cpu().tolist())
                            recent_lengths.extend(episode_lengths[done].cpu().tolist())
                            counters["episodes"] += int(done.sum().item())
                            episode_returns[done] = 0
                            episode_lengths[done] = 0
                        # These are reporting-only paths; they never alter obs.
                        for name, event in info.get("events", {}).items():
                            count = float(torch.as_tensor(event).sum().item())
                            counters[name] += count
                            update_events[name] += count
                        for name, metric in info.get("episode", {}).items():
                            metric = torch.as_tensor(metric)
                            selected = metric[done] if metric.ndim and metric.shape[0] == num_envs else metric
                            if selected.numel():
                                episode_diagnostics.setdefault(name, []).append(float(selected.float().mean().item()))
                        obs, hidden, episode_start = next_obs, next_hidden, done
                    global_step += num_envs
                    if config.callback_every_steps > 0 and (step + 1) % config.callback_every_steps == 0:
                        notify("rollout_step", {
                            "update": current_update, "rollout_step": step + 1,
                            "reward_mean": float(raw_reward_sum.item() / ((step + 1) * num_envs)),
                            "event_counts": dict(counters),
                        })
                with torch.no_grad():
                    last_value = policy.value(obs, hidden, episode_start)
                    advantages, returns = compute_gae(
                        rewards, values, dones, last_value, config.gamma, config.gae_lambda
                    )
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                rollout_seconds = time.perf_counter() - rollout_start
                chunk_length = config.sequence_length
                sequences = steps // chunk_length * num_envs
                image_chunks = _chunks(images, chunk_length)
                proprio_chunks = _chunks(proprio, chunk_length)
                start_chunks = _chunks(starts, chunk_length)
                action_chunks = _chunks(raw_actions, chunk_length)
                old_log_prob_chunks = _chunks(log_probs, chunk_length)
                old_value_chunks = _chunks(values, chunk_length)
                advantage_chunks = _chunks(advantages, chunk_length)
                return_chunks = _chunks(returns, chunk_length)
                initial_hidden = hidden_states[::chunk_length].reshape(sequences, -1)
                policy.train()
                losses: list[tuple[float, ...]] = []
                early_stop = False
                update_start = time.perf_counter()
                for epoch in range(config.update_epochs):
                    permutation = torch.randperm(sequences, device=device)
                    epoch_kls = []
                    for offset in range(0, sequences, config.minibatch_sequences):
                        batch = permutation[offset:offset + config.minibatch_sequences]
                        new_log_prob, entropy, new_value = policy.evaluate_sequence(
                            {"image": image_chunks[:, batch], "proprio": proprio_chunks[:, batch]},
                            initial_hidden[batch], start_chunks[:, batch], action_chunks[:, batch],
                        )
                        log_ratio = new_log_prob - old_log_prob_chunks[:, batch]
                        ratio = log_ratio.exp()
                        advantage = advantage_chunks[:, batch]
                        policy_loss = torch.maximum(
                            -advantage * ratio,
                            -advantage * ratio.clamp(1.0 - config.clip_coef, 1.0 + config.clip_coef),
                        ).mean()
                        old_value = old_value_chunks[:, batch]
                        clipped_value = old_value + (new_value - old_value).clamp(
                            -config.clip_coef, config.clip_coef
                        )
                        target = return_chunks[:, batch]
                        value_loss = 0.5 * torch.maximum(
                            (new_value - target).square(), (clipped_value - target).square()
                        ).mean()
                        entropy_mean = entropy.mean()
                        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite PPO loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                        if not torch.isfinite(grad_norm):
                            raise FloatingPointError("non-finite PPO gradient")
                        optimizer.step()
                        with torch.no_grad():
                            approximate_kl = ((ratio - 1) - log_ratio).mean().item()
                            clip_fraction = ((ratio - 1).abs() > config.clip_coef).float().mean().item()
                        epoch_kls.append(approximate_kl)
                        losses.append((
                            policy_loss.item(), value_loss.item(), entropy_mean.item(),
                            approximate_kl, clip_fraction, grad_norm.item(),
                        ))
                    if config.target_kl > 0 and float(np.mean(epoch_kls)) > config.target_kl:
                        early_stop = True
                        break
                update_seconds = time.perf_counter() - update_start
                elapsed = time.perf_counter() - start_time
                means = np.mean(losses, axis=0)
                return_variance = returns.var(unbiased=False)
                explained_variance = (
                    1 - (returns - values).var(unbiased=False) / return_variance
                ).item() if return_variance.item() > 1e-12 else None
                last_metrics = {
                    "update": current_update, "global_step": global_step,
                    "device": str(device), "gpu_training": device.type == "cuda",
                    "num_envs": num_envs, "elapsed_seconds": elapsed,
                    "samples_per_second": (global_step - initial_step) / max(elapsed, 1e-6),
                    "rollout_samples_per_second": steps * num_envs / max(rollout_seconds, 1e-6),
                    "rollout_seconds": rollout_seconds, "update_seconds": update_seconds,
                    "reward_mean": float(raw_reward_sum.item() / (steps * num_envs)),
                    "episode_return_mean_100": float(np.mean(recent_returns)) if recent_returns else None,
                    "episode_length_mean_100": float(np.mean(recent_lengths)) if recent_lengths else None,
                    "policy_loss": float(means[0]), "value_loss": float(means[1]),
                    "gaussian_entropy": float(means[2]), "approximate_kl": float(means[3]),
                    "clip_fraction": float(means[4]), "gradient_norm": float(means[5]),
                    "explained_variance": explained_variance, "kl_early_stop": early_stop,
                    "epochs_completed": epoch + 1, "event_counts": dict(counters),
                    "update_event_counts": dict(update_events),
                    "episode_metrics": {name: float(np.mean(items)) for name, items in episode_diagnostics.items()},
                }
                if device.type == "cuda":
                    last_metrics["gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated(device)
                    last_metrics["gpu_memory_reserved_bytes"] = torch.cuda.memory_reserved(device)
                metrics_file.write(json.dumps(last_metrics, allow_nan=False) + "\n")
                notify("update", last_metrics)
                print(json.dumps(last_metrics, allow_nan=False), flush=True)
                if (
                    current_update % config.checkpoint_every_updates == 0
                    or time.perf_counter() - last_checkpoint_time >= config.checkpoint_every_seconds
                ):
                    save_checkpoint(f"checkpoint_{global_step:012d}.pt")
                if config.max_walltime_seconds is not None and elapsed >= config.max_walltime_seconds:
                    last_metrics["stop_reason"] = "max_walltime_seconds"
                    break
        except KeyboardInterrupt:
            last_metrics["stop_reason"] = "keyboard_interrupt"
        finally:
            final_path = save_checkpoint("checkpoint_final.pt")
    notify("complete", {**last_metrics, "checkpoint": str(final_path)})
    return final_path
