"""Visual GRU behavior cloning from independently validated expert trajectories.

Only image/proprioception observations and action targets enter the loss.
The collector's valid_environment mask selects complete successful trajectories;
its report and evaluator state are never used as network features or targets.
This produces weights for PPO --init-checkpoint, not a resumable PPO run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from .policy import RecurrentActorCritic
from .progress import update


def _load_dataset(path, *, mmap=False):
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    for name in ("image", "proprio", "action", "episode_start", "valid_environment", "valid_transition"):
        if not isinstance(data.get(name), torch.Tensor):
            raise ValueError(f"Dataset requires tensor {name}")
    image = data["image"]
    if image.ndim != 5 or image.shape[2] != 5 or min(image.shape[3:]) < 32:
        raise ValueError("image must have shape [T,N,5,H,W], with H,W >= 32")
    steps, worlds = image.shape[:2]
    if min(steps, worlds) < 1:
        raise ValueError("Dataset must contain at least one step and environment")
    proprio = data["proprio"]
    if proprio.ndim != 3 or tuple(proprio.shape[:2]) != (steps, worlds) or proprio.shape[-1] not in (67, 68):
        raise ValueError("proprio must have shape [T,N,67] or [T,N,68]")
    for name, shape in (("action", (steps, worlds, 19)),
                        ("episode_start", (steps, worlds)),
                        ("valid_transition", (steps, worlds)),
                        ("valid_environment", (worlds,))):
        if tuple(data[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if any(not data[name].is_floating_point() for name in ("image", "proprio", "action")):
        raise ValueError("image, proprio and action must be floating-point tensors")
    if any(data[name].dtype != torch.bool for name in ("episode_start", "valid_environment", "valid_transition")):
        raise ValueError("episode_start, valid_environment and valid_transition must be boolean tensors")
    valid = torch.nonzero(data["valid_environment"], as_tuple=False).flatten()
    if not len(valid):
        raise ValueError("No valid_environment=true trajectories: behavior cloning requires complete successful, physically valid demonstrations")
    transitions = data["valid_transition"][:, valid]
    if not bool(transitions[0].all()) or bool((~transitions[:-1] & transitions[1:]).any()):
        raise ValueError("Each valid environment must have a nonempty contiguous prefix of valid_transition steps before terminal padding")
    if not isinstance(data.get("environment"), dict):
        raise ValueError("Dataset requires an environment configuration dictionary")
    expected_proprio = 68 if data["environment"].get("observe_time", False) else 67
    if proprio.shape[-1] != expected_proprio:
        raise ValueError("Dataset proprio dimension disagrees with environment.observe_time")
    return data, valid


def _batch(data, sequences, sequence_length, device):
    worlds, starts = sequences[:, 0], sequences[:, 1]
    time_indices = starts[None] + torch.arange(sequence_length)[:, None]
    mask = time_indices < data["image"].shape[0]
    time_indices = time_indices.clamp_max(data["image"].shape[0] - 1)
    mask &= data["valid_transition"][time_indices, worlds[None]]
    result = {name: data[name][time_indices, worlds[None]].to(device)
              for name in ("image", "proprio", "action", "episode_start")}
    mask = mask.to(device)
    for name in ("image", "proprio", "action"):
        broadcast = mask.reshape(*mask.shape, *((1,) * (result[name].ndim - 2)))
        # Reset/padding states can contain arbitrary values and never enter
        # either the visual encoder or the regression target.
        result[name] = result[name].masked_fill(~broadcast, 0)
        if not bool(torch.isfinite(result[name]).all()):
            raise ValueError(f"Nonfinite {name} in a selected valid trajectory")
    if bool((result["action"].abs() > 1.000001).any()):
        raise ValueError("Expert actions must be in [-1,1]")
    result["episode_start"] |= ~mask
    result["mask"] = mask
    return result


def _sequence_loss(policy, batch, leg_only=False):
    steps, count = batch["action"].shape[:2]
    encoded = policy._encode(
        batch["image"].reshape(steps * count, *batch["image"].shape[2:]),
        batch["proprio"].reshape(steps * count, policy.proprio_dim),
    ).reshape(steps, count, -1)
    hidden = policy.initial_hidden(count, encoded.device)
    errors = []
    for step in range(steps):
        hidden = hidden * (~batch["episode_start"][step]).to(hidden.dtype)[:, None]
        hidden = policy.memory(encoded[step], hidden)
        predicted = policy.actor(hidden).tanh()
        error = (predicted - batch["action"][step]).square()
        errors.append((error[:, :12] if leg_only else error).mean(-1))
    mask = batch["mask"]
    return (torch.stack(errors) * mask).sum() / mask.sum(), int(mask.sum().item())


@torch.no_grad()
def _training_mse(policy, data, sequences, sequence_length, batch_size, device,
                  startup_steps=None, leg_only=False):
    policy.eval()
    total, samples = 0., 0
    for begin in range(0, len(sequences), batch_size):
        batch = _batch(data, sequences[begin:begin + batch_size], sequence_length, device)
        if startup_steps is not None:
            positions = (sequences[begin:begin + batch_size, 1][None]
                         + torch.arange(sequence_length)[:, None]).to(device)
            batch["mask"] &= positions < startup_steps
        loss, count = _sequence_loss(policy, batch, leg_only=leg_only)
        total += float(loss) * count
        samples += count
    return total / samples


def _initialize_policy(policy, state):
    """Permit only an exact architecture match or one zero-initialized clock column."""
    source_kwargs = state["network_kwargs"]
    target_kwargs = policy.network_kwargs()
    padded_clock = False
    weights = state["model"]
    if source_kwargs != target_kwargs:
        extended_kwargs = dict(source_kwargs)
        extended_kwargs["proprio_dim"] = 68
        if source_kwargs.get("proprio_dim") != 67 or target_kwargs["proprio_dim"] != 68 or extended_kwargs != target_kwargs:
            raise ValueError("Initialization checkpoint architecture does not match demonstration observations")
        name = "proprioception.0.weight"
        source_weight = weights[name]
        if tuple(source_weight.shape) != (policy.proprioception[0].out_features, 67):
            raise ValueError("67-D initialization requires the original 67-column proprioception weight")
        weights = dict(weights)
        weights[name] = torch.cat((source_weight, source_weight.new_zeros((source_weight.shape[0], 1))), dim=1)
        padded_clock = True
    policy.load_state_dict(weights, strict=True)
    return padded_clock


def train_behavior_clone(dataset, run_dir, *, init_checkpoint=None, epochs=10,
                         sequence_length=32, batch_size=4, learning_rate=3e-4,
                         device="cuda:0", seed=42, ignore_previous_action=False,
                         startup_repeats=1, startup_steps=400):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if sequence_length not in (32, 64):
        raise ValueError("sequence_length must be 32 or 64")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (startup_repeats, startup_steps)):
        raise ValueError("startup_repeats and startup_steps must be positive integers")
    started = time.perf_counter()
    data, valid = _load_dataset(dataset)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU behavior cloning cannot start")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    steps, worlds = data["image"].shape[:2]
    image_shape = tuple(data["image"].shape[2:])
    proprio_dim = data["proprio"].shape[-1]
    policy = RecurrentActorCritic(image_shape=image_shape, proprio_dim=proprio_dim).to(device)
    padded_clock = False
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location=device, weights_only=False)
        padded_clock = _initialize_policy(policy, state)
    if ignore_previous_action:
        # Keep all input columns in place. The saved zero columns also
        # remove this feature during ordinary evaluation, without an eval flag.
        input_weight = policy.proprioception[0].weight
        gradient_mask = torch.ones_like(input_weight)
        gradient_mask[:, 47:66] = 0
        with torch.no_grad():
            input_weight[:, 47:66].zero_()
        input_weight.register_hook(lambda gradient: gradient * gradient_mask)
    # BC does not fit value targets. PPO creates trainable parameters again
    # when loading this state_dict, including its exploration standard deviation.
    policy.critic.requires_grad_(False)
    policy.log_std.requires_grad_(False)
    with torch.no_grad():
        policy.log_std.fill_(-2.)
    optimizer = torch.optim.Adam((parameter for parameter in policy.parameters()
                                  if parameter.requires_grad), lr=learning_rate)
    valid_steps = data["valid_transition"][:, valid].sum(0)
    sequences = torch.tensor([(int(world), begin) for world, length in zip(valid, valid_steps)
                              for begin in range(0, int(length), sequence_length)], dtype=torch.long)
    startup_sequences = sequences[sequences[:, 1] < startup_steps]
    sampled_sequences = torch.cat((sequences, startup_sequences.repeat(startup_repeats - 1, 1)))
    unique_valid_steps = int(valid_steps.sum())
    environment = dict(data["environment"])
    for name, value in (("image_width", image_shape[2]), ("image_height", image_shape[1])):
        if name in environment and environment[name] != value:
            raise ValueError(f"Dataset environment {name} disagrees with image tensor")
        environment[name] = value
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint_final.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Use a new run directory; checkpoint already exists: {checkpoint}")
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    for name in ("latest_world.png", "latest_rgb.png", "latest_depth.png", "preview_frame.json"):
        source, target = Path(dataset).parent / name, run_dir / name
        if source.is_file() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
    optimization = "GPU 网络优化" if device.type == "cuda" else "CPU 网络优化"
    preview_detail = f"画面是最后验证的专家帧；当前为{optimization}，设备={device}。"
    update("视觉行为克隆初始化", f"epoch 0/{epochs}，正在计算初始 MSE。{preview_detail}",
           run_dir=run_dir, metrics={"epoch": 0, "epochs": epochs, "optimization_mse": None})
    initial_mse = _training_mse(policy, data, sequences, sequence_length, batch_size, device)
    optimizer_steps = 0
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            policy.train()
            shuffled = sampled_sequences[torch.randperm(len(sampled_sequences))]
            loss_total, samples = 0., 0
            for begin in range(0, len(shuffled), batch_size):
                batch = _batch(data, shuffled[begin:begin + batch_size], sequence_length, device)
                optimizer.zero_grad(set_to_none=True)
                loss, count = _sequence_loss(policy, batch)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite behavior cloning loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in policy.parameters() if parameter.requires_grad], 1.)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError("Nonfinite behavior cloning gradient")
                optimizer.step()
                optimizer_steps += 1
                loss_total += float(loss.detach()) * count
                samples += count
            metrics = {"epoch": epoch + 1, "optimization_mse": loss_total / samples,
                       "optimizer_steps": optimizer_steps, "transitions": samples,
                       "unique_valid_steps": unique_valid_steps, "sampled_steps": samples}
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(json.dumps(metrics), flush=True)
            update("视觉行为克隆训练", f"epoch {epoch + 1}/{epochs}，MSE={metrics['optimization_mse']:.6f}。{preview_detail}",
                   run_dir=run_dir, metrics={**metrics, "epochs": epochs})
    final_mse = _training_mse(policy, data, sequences, sequence_length, batch_size, device)
    startup_mse = _training_mse(policy, data, startup_sequences, sequence_length, batch_size,
                                device, startup_steps=startup_steps)
    startup_leg_mse = _training_mse(policy, data, startup_sequences, sequence_length, batch_size,
                                    device, startup_steps=startup_steps, leg_only=True)
    report = {
        "dataset": str(dataset), "initialization_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "device": str(device), "gpu_training": device.type == "cuda", "seed": seed,
        "source_environments": worlds, "valid_environments": len(valid),
        "excluded_environments": worlds - len(valid), "steps_per_trajectory": steps,
        "valid_steps_per_trajectory": valid_steps.tolist(),
        "training_transitions": unique_valid_steps, "epochs": epochs,
        "unique_valid_steps": unique_valid_steps, "sampled_steps_per_epoch": samples,
        "startup_repeats": startup_repeats, "startup_steps": startup_steps,
        "startup_sampling": "total repetition count for whole sequences starting before startup_steps",
        "sequence_length": sequence_length, "batch_size": batch_size,
        "learning_rate": learning_rate, "optimizer_steps": optimizer_steps,
        "initial_training_mse": initial_mse, "final_training_mse": final_mse,
        "startup_mse": startup_mse, "startup_leg_mse": startup_leg_mse,
        "mse_scope": "unrepeated valid transitions; startup metrics cover only positions < startup_steps",
        "hidden_state": "zero at truncated sequence start; reset at episode_start",
        "actor_inputs": ["image", "proprio"], "target": "normalized expert action",
        "proprio_dim": proprio_dim,
        "clock": {"enabled": proprio_dim == 68, "column": 67 if proprio_dim == 68 else None,
                  "formula": "episode_steps * control_dt / episode_seconds",
                  "unit": "dimensionless episode fraction", "source": "robot-owned episode control clock",
                  "scope": "time conditioning does not establish unloading generalization"},
        "initialization_zero_padded_clock_column": padded_clock,
        "initialization_weight_change": "append one zero column to proprioception.0.weight; all other weights loaded strictly" if padded_clock else None,
        "ignore_previous_action": ignore_previous_action,
        "masked_proprio_columns": [47, 66] if ignore_previous_action else None,
        "masked_proprio_columns_semantics": "start inclusive, stop exclusive; zero weights and gradients during BC only",
        "critic": "weights preserved; no value targets fitted",
        "log_std": -2., "elapsed_wall_s": time.perf_counter() - started,
        "scope": "behavior cloning training loss only; not held-out unloading success",
    }
    state = {
        "format_version": 1, "checkpoint_kind": "behavior_clone_initialization",
        "network_kwargs": policy.network_kwargs(),
        "model": {name: value.detach().cpu() for name, value in policy.state_dict().items()},
        "environment": environment, "global_step": 0,
        "bc_optimizer_steps": optimizer_steps, "metrics": report,
        "initialization_checkpoint": str(init_checkpoint) if init_checkpoint else None,
    }
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(checkpoint)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    update("视觉行为克隆权重已保存", f"epoch {epochs}/{epochs}，最终训练 MSE={final_mse:.6f}。{preview_detail}",
           run_dir=run_dir, metrics={"epoch": epochs, "epochs": epochs, "training_mse": final_mse,
                                     "optimizer_steps": optimizer_steps, "checkpoint": str(checkpoint)})
    print(json.dumps({"checkpoint": str(checkpoint), **report}, indent=2), flush=True)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, choices=(32, 64), default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-previous-action", action="store_true")
    def positive_integer(value):
        value = int(value)
        if value < 1:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return value
    parser.add_argument("--startup-repeats", type=positive_integer, default=1)
    parser.add_argument("--startup-steps", type=positive_integer, default=400)
    train_behavior_clone(**vars(parser.parse_args()))
