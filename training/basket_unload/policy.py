"""RGB-D and proprioception recurrent policy for the isolated unload task.

Only the two observation tensors enter this network. Object poses, contact
labels, rewards and environment diagnostics are deliberately outside its API.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F


class RecurrentActorCritic(nn.Module):
    """Small visual encoder with a shared GRU and independent policy/value heads.

    ``act`` returns an *unsquashed* Gaussian sample. The environment receives
    ``raw_action.tanh()``. Storing the raw sample avoids numerical inversion at
    the ends of the bounded action interval during PPO updates.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (5, 96, 128),
        proprio_dim: int = 67,
        action_dim: int = 19,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if len(image_shape) != 3 or image_shape[0] != 5 or min(image_shape[1:]) < 32:
            raise ValueError("image_shape must be (5, H, W), with H and W >= 32")
        self.image_shape = tuple(image_shape)
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.visual = nn.Sequential(
            nn.Conv2d(5, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
            nn.Linear(32 * 3 * 4, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.proprioception = nn.Sequential(
            nn.Linear(proprio_dim, 64), nn.LayerNorm(64), nn.SiLU()
        )
        self.memory = nn.GRUCell(192, hidden_size)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def network_kwargs(self) -> dict:
        return {
            "image_shape": self.image_shape,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.hidden_size,
        }

    def initial_hidden(self, batch_size: int, device: torch.device | str | None = None) -> Tensor:
        return torch.zeros(
            batch_size,
            self.hidden_size,
            device=device if device is not None else self.actor.weight.device,
        )

    def _encode(self, image: Tensor, proprio: Tensor) -> Tensor:
        return torch.cat(
            (self.visual(image.float()), self.proprioception(proprio.float())), dim=-1
        )

    def _distribution(self, hidden: Tensor) -> Normal:
        # Bounds prevent exponent overflow while retaining learned exploration.
        std = self.log_std.clamp(-5.0, 1.0).exp().expand(hidden.shape[0], -1)
        return Normal(self.actor(hidden), std)

    @staticmethod
    def _log_prob(distribution: Normal, raw_action: Tensor) -> Tensor:
        # Stable log(1 - tanh(x)^2), including highly saturated raw actions.
        log_jacobian = 2.0 * (math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
        return (distribution.log_prob(raw_action) - log_jacobian).sum(dim=-1)

    def _step(self, obs: Mapping[str, Tensor], hidden: Tensor, episode_start: Tensor) -> Tensor:
        hidden = hidden * (~episode_start.bool()).to(hidden.dtype).unsqueeze(-1)
        return self.memory(self._encode(obs["image"], obs["proprio"]), hidden)

    def act(
        self,
        obs: Mapping[str, Tensor],
        hidden: Tensor,
        episode_start: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return raw_action, squashed-action log probability, value, next_hidden."""
        next_hidden = self._step(obs, hidden, episode_start)
        distribution = self._distribution(next_hidden)
        raw_action = distribution.mean if deterministic else distribution.sample()
        return (
            raw_action,
            self._log_prob(distribution, raw_action),
            self.critic(next_hidden).squeeze(-1),
            next_hidden,
        )

    def value(self, obs: Mapping[str, Tensor], hidden: Tensor, episode_start: Tensor) -> Tensor:
        return self.critic(self._step(obs, hidden, episode_start)).squeeze(-1)

    def evaluate_sequence(
        self,
        obs: Mapping[str, Tensor],
        hidden: Tensor,
        episode_start: Tensor,
        raw_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Replay a [time, batch] chunk with truncated backpropagation through time.

        The supplied hidden state is the detached rollout state immediately
        before the chunk. Episode boundaries inside the chunk reset memory.
        Entropy is that of the underlying Gaussian, a consistent exploration
        regularizer; action likelihoods include the tanh transform Jacobian.
        """
        steps, batches = raw_action.shape[:2]
        encoded = self._encode(
            obs["image"].reshape(steps * batches, *obs["image"].shape[2:]),
            obs["proprio"].reshape(steps * batches, -1),
        ).reshape(steps, batches, -1)
        states = []
        for step in range(steps):
            hidden = hidden * (~episode_start[step].bool()).to(hidden.dtype).unsqueeze(-1)
            hidden = self.memory(encoded[step], hidden)
            states.append(hidden)
        memory = torch.stack(states)
        distribution = self._distribution(memory.reshape(steps * batches, -1))
        log_prob = self._log_prob(distribution, raw_action.reshape(steps * batches, -1))
        entropy = distribution.entropy().sum(-1)
        value = self.critic(memory).squeeze(-1)
        return log_prob.reshape(steps, batches), entropy.reshape(steps, batches), value
