"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        dims = (state_dim, *hidden_dims, chunk_size * action_dim)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def _predict_action_chunk(self, state: torch.Tensor) -> torch.Tensor:
        pred = self.network(state)
        return pred.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_chunk = self._predict_action_chunk(state)
        return nn.functional.mse_loss(pred_chunk, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        del num_steps
        return self._predict_action_chunk(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.chunk_dim = chunk_size * action_dim

        dims = (state_dim + self.chunk_dim + 1, *hidden_dims, self.chunk_dim)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def _predict_velocity(
        self,
        state: torch.Tensor,
        noisy_action_chunk: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        flat_chunk = noisy_action_chunk.view(-1, self.chunk_dim)
        model_input = torch.cat((state, flat_chunk, t), dim=-1)
        pred = self.network(model_input)
        return pred.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(action_chunk)
        t = torch.rand(action_chunk.shape[0], 1, device=action_chunk.device)
        x_t = (1.0 - t[:, None, :]) * noise + t[:, None, :] * action_chunk
        target_velocity = action_chunk - noise
        pred_velocity = self._predict_velocity(state, x_t, t)
        return nn.functional.mse_loss(pred_velocity, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        action_chunk = torch.randn(
            state.shape[0],
            self.chunk_size,
            self.action_dim,
            device=state.device,
            dtype=state.dtype,
        )
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full(
                (state.shape[0], 1),
                step / num_steps,
                device=state.device,
                dtype=state.dtype,
            )
            velocity = self._predict_velocity(state, action_chunk, t)
            action_chunk = action_chunk + dt * velocity

        return action_chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
