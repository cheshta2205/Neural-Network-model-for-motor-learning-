"""Abstract trainer utilities for RL algorithms.

This base class centralizes the shared mechanics that most trainers need:

- model registration and device placement
- optimizer registration and lifecycle helpers
- gradient zeroing, clipping, and stepping

Algorithm-specific update logic remains the responsibility of subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Optional

import torch
from torch import nn


class BaseTrainer(ABC):
    """Base class for RL trainers with shared optimizer/model utilities."""

    def __init__(
        self,
        models: Optional[Mapping[str, nn.Module]] = None,
        optimizers: Optional[Mapping[str, torch.optim.Optimizer]] = None,
        device: Optional[torch.device | str] = None,
        grad_clip_norm: Optional[float] = None,
    ) -> None:
        self.models: dict[str, nn.Module] = dict(models or {})
        self.optimizers: dict[str, torch.optim.Optimizer] = dict(optimizers or {})
        self.device = torch.device(device) if device is not None else None
        self.grad_clip_norm = grad_clip_norm

        if self.device is not None:
            self.to(self.device)

    def register_model(self, name: str, model: nn.Module) -> nn.Module:
        """Register a model and move it to the trainer device if available."""
        self.models[name] = model
        if self.device is not None:
            model.to(self.device)
        return model

    def register_optimizer(self, name: str, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Register an optimizer under a stable name."""
        self.optimizers[name] = optimizer
        return optimizer

    def to(self, device: torch.device | str) -> "BaseTrainer":
        """Move all registered models to a device."""
        self.device = torch.device(device)
        for model in self.models.values():
            model.to(self.device)
        return self

    def train(self) -> None:
        """Switch all registered models to train mode."""
        for model in self.models.values():
            model.train()

    def eval(self) -> None:
        """Switch all registered models to eval mode."""
        for model in self.models.values():
            model.eval()

    def zero_grad(self, optimizer_names: Optional[Iterable[str]] = None) -> None:
        """Zero gradients for all or a selected subset of optimizers."""
        for optimizer in self._iter_optimizers(optimizer_names):
            optimizer.zero_grad()

    def step(self, optimizer_names: Optional[Iterable[str]] = None) -> None:
        """Step all or a selected subset of optimizers."""
        for optimizer in self._iter_optimizers(optimizer_names):
            optimizer.step()

    def clip_grad_norm_(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None,
        max_norm: Optional[float] = None,
    ) -> None:
        """Clip gradients on a parameter iterable or on all registered models."""
        clip_value = self.grad_clip_norm if max_norm is None else max_norm
        if clip_value is None:
            return

        if parameters is not None:
            torch.nn.utils.clip_grad_norm_(list(parameters), clip_value)
            return

        for model in self.models.values():
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    def optimize(
        self,
        loss: torch.Tensor,
        optimizer_names: Optional[Iterable[str]] = None,
        *,
        retain_graph: bool = False,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None,
        max_norm: Optional[float] = None,
    ) -> torch.Tensor:
        """Convenience helper for a standard backward/update step.

        Subclasses can call this when they only need a simple backward pass.
        More specialized algorithms may still manage their own schedules and
        call `zero_grad`, `step`, or `clip_grad_norm_` directly.
        """
        self.zero_grad(optimizer_names)
        loss.backward(retain_graph=retain_graph)
        self.clip_grad_norm_(parameters=parameters, max_norm=max_norm)
        self.step(optimizer_names)
        return loss

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Run one algorithm-specific training update."""

    def _iter_optimizers(self, optimizer_names: Optional[Iterable[str]] = None):
        if optimizer_names is None:
            yield from self.optimizers.values()
            return

        for name in optimizer_names:
            yield self.optimizers[name]