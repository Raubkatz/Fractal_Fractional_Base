"""
class_b_fractional_optimizers_OHA_torch.py

Herrera-style fractional optimizers for PyTorch (tested with PyTorch 2.15).

This file is the PyTorch counterpart of
`class_b_tf_fractional_optimizers_OHA_keras3.py`. Same classes, same
arguments, same update rules, written as `torch.optim.Optimizer` subclasses.

    ┌──────────────────────────────────────────────────────────────────────┐
    │ ORIGINAL WORK: the Herrera-style fractional optimizers were developed │
    │ by Oscar Herrera-Alcántara and co-authors. If you use them, cite:     │
    │  [1] O. Herrera-Alcántara, Applied Sciences 12(18), 9264 (2022),      │
    │      https://doi.org/10.3390/app12189264                              │
    │  [2] O. Herrera-Alcántara, J. R. Castelán-Aguilar, Fractal and        │
    │      Fractional 7(7), 500 (2023),                                     │
    │      https://doi.org/10.3390/fractalfract7070500                      │
    │ This file is an independent re-implementation, not the authors' code.│
    └──────────────────────────────────────────────────────────────────────┘

Main idea
---------
For a parameter tensor w and ordinary gradient g = dL/dw, the modified gradient is

    g^(nu) = g * f_nu(w),    f_nu(w) = (|w| + epsilon)^(1 - nu) / Gamma(2 - nu),

where nu is the fractional order (`vderiv`), epsilon > 0 avoids the
singularity at w = 0, and Gamma is the Gamma function. Only the gradient
input of the base optimizer changes; there is no memory. At nu = 1,
f_nu(w) = 1 and each optimizer reduces to its base optimizer.

Implemented optimizers
----------------------
    FSGD        (SGD, optional momentum / Nesterov)
    FAdam       (Adam, optional AMSGrad)
    FAdagrad    (Adagrad)
    FAdadelta   (Adadelta)
    FRMSprop    (RMSprop, optional momentum / centered)
    FAdamW      (AdamW, decoupled weight decay applied after the Adam step)

PyTorch conventions
-------------------
- `lr=` is the learning rate; `learning_rate=` is accepted as an alias so
  that settings can be copied from the Keras version.
- All other argument names (beta_1, beta_2, epsilon, rho, vderiv, ...) are the
  same as in the Keras version, so configurations transfer one to one.
  (They differ from PyTorch's own `betas=` / `eps=` on purpose.)
- Every hyperparameter lives in the parameter group, so different groups
  can use different settings, and `torch.optim.lr_scheduler` works as usual.
- `optimizer.state_dict()` / `load_state_dict()` save and restore all state.
- Sparse gradients (e.g. nn.Embedding(sparse=True)) are densified, as in the
  Keras version.
- The step counter is kept per parameter (`state["step"]`, the PyTorch
  convention). It equals Keras' global `optimizer.iterations + 1` whenever
  every parameter receives a gradient in every step.
- Gradient clipping is done outside the optimizer in PyTorch, e.g.
  `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before `step()`.

Usage
-----
    from class_b_fractional_optimizers_OHA_torch import FAdam

    optimizer = FAdam(model.parameters(), lr=1e-3, vderiv=0.9, frac_epsilon=1e-6)
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional

import torch


# =============================================================================
# Small helpers
# =============================================================================


def _resolve_lr(lr, learning_rate) -> float:
    """`learning_rate=` is an alias for `lr=` (Keras naming)."""
    value = lr if learning_rate is None else learning_rate
    if isinstance(value, torch.Tensor):
        value = float(value.item())
    return float(value)


def _dense(grad: torch.Tensor) -> torch.Tensor:
    """Densify sparse gradients (same behaviour as the Keras version)."""
    return grad.to_dense() if grad.is_sparse else grad


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# =============================================================================
# Shared helper base
# =============================================================================


class _FractionalHerreraBase:
    """Shared Herrera-style gradient scaling and argument checks."""

    @staticmethod
    def _check_fractional(group: Dict[str, Any]) -> None:
        vderiv = group["vderiv"]
        if not isinstance(vderiv, (float, int)) or isinstance(vderiv, bool):
            raise TypeError("`vderiv` must be a float.")
        _check(0.0 < float(vderiv) < 2.0, "`vderiv` must satisfy 0 < vderiv < 2.")
        _check(group["frac_epsilon"] > 0.0, "`frac_epsilon` must be > 0.")
        _check(group["lr"] >= 0.0, "`lr` must be >= 0.")

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        super().add_param_group(param_group)
        self._validate_group(self.param_groups[-1])

    @staticmethod
    def _fractional_factor(p: torch.Tensor, vderiv: float, frac_epsilon: float) -> torch.Tensor:
        """f_nu(w) = (|w| + eps)^(1 - nu) / Gamma(2 - nu), element-wise."""
        gamma_factor = math.gamma(2.0 - float(vderiv))
        return torch.pow(p.abs() + frac_epsilon, 1.0 - float(vderiv)) / gamma_factor

    def _fractional_gradient(self, grad: torch.Tensor, p: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        grad = _dense(grad).to(p.dtype)
        return grad * self._fractional_factor(p, group["vderiv"], group["frac_epsilon"])


# =============================================================================
# FSGD
# =============================================================================


class FSGD(_FractionalHerreraBase, torch.optim.Optimizer):
    """
    Herrera-style fractional SGD.

    Without momentum:   w_{t+1} = w_t - lr * g_t^(nu)
    With momentum:      m_t = momentum * m_{t-1} - lr * g_t^(nu);  w_{t+1} = w_t + m_t
    Nesterov:           w_{t+1} = w_t + momentum * m_t - lr * g_t^(nu)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        learning_rate: Optional[float] = None,
    ):
        defaults = dict(
            lr=_resolve_lr(lr, learning_rate),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            vderiv=vderiv,
            frac_epsilon=float(frac_epsilon),
        )
        super().__init__(params, defaults)

    def _validate_group(self, group: Dict[str, Any]) -> None:
        self._check_fractional(group)
        _check(0.0 <= group["momentum"] < 1.0, "`momentum` must satisfy 0 <= momentum < 1.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1

                grad = self._fractional_gradient(p.grad, p, group)

                if mu > 0.0:
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    m = state["momentum"]
                    new_m = mu * m - lr * grad
                    m.copy_(new_m)
                    if group["nesterov"]:
                        p.add_(mu * new_m - lr * grad)
                    else:
                        p.add_(new_m)
                else:
                    p.sub_(lr * grad)
        return loss


# =============================================================================
# FAdam
# =============================================================================


class FAdam(_FractionalHerreraBase, torch.optim.Optimizer):
    """
    Herrera-style fractional Adam.

        m_t = beta_1 m_{t-1} + (1 - beta_1) g_t^(nu)
        v_t = beta_2 v_{t-1} + (1 - beta_2) (g_t^(nu))^2
        w_{t+1} = w_t - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)

    with m_hat_t = m_t / (1 - beta_1^t), v_hat_t = v_t / (1 - beta_2^t)
    (running maximum of v_t for AMSGrad).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        learning_rate: Optional[float] = None,
    ):
        defaults = dict(
            lr=_resolve_lr(lr, learning_rate),
            beta_1=float(beta_1),
            beta_2=float(beta_2),
            epsilon=float(epsilon),
            amsgrad=bool(amsgrad),
            vderiv=vderiv,
            frac_epsilon=float(frac_epsilon),
        )
        super().__init__(params, defaults)

    def _validate_group(self, group: Dict[str, Any]) -> None:
        self._check_fractional(group)
        _check(0.0 <= group["beta_1"] < 1.0, "`beta_1` must satisfy 0 <= beta_1 < 1.")
        _check(0.0 <= group["beta_2"] < 1.0, "`beta_2` must satisfy 0 <= beta_2 < 1.")
        _check(group["epsilon"] > 0.0, "`epsilon` must be > 0.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, b1, b2, eps = group["lr"], group["beta_1"], group["beta_2"], group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        state["vhat"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]

                grad = self._fractional_gradient(p.grad, p, group)

                m, v = state["m"], state["v"]
                new_m = b1 * m + (1.0 - b1) * grad
                new_v = b2 * v + (1.0 - b2) * grad * grad
                m.copy_(new_m)
                v.copy_(new_v)

                m_hat = new_m / (1.0 - b1 ** t)
                if group["amsgrad"]:
                    vhat = state["vhat"]
                    new_vhat = torch.maximum(vhat, new_v)
                    vhat.copy_(new_vhat)
                    v_hat = new_vhat / (1.0 - b2 ** t)
                else:
                    v_hat = new_v / (1.0 - b2 ** t)

                p.sub_(lr * m_hat / (v_hat.sqrt() + eps))
        return loss


# =============================================================================
# FAdagrad
# =============================================================================


class FAdagrad(_FractionalHerreraBase, torch.optim.Optimizer):
    """
    Herrera-style fractional Adagrad.

        acc_t = acc_{t-1} + (g_t^(nu))^2
        w_{t+1} = w_t - lr * g_t^(nu) / (sqrt(acc_t) + epsilon)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.001,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        learning_rate: Optional[float] = None,
    ):
        defaults = dict(
            lr=_resolve_lr(lr, learning_rate),
            initial_accumulator_value=float(initial_accumulator_value),
            epsilon=float(epsilon),
            vderiv=vderiv,
            frac_epsilon=float(frac_epsilon),
        )
        super().__init__(params, defaults)

    def _validate_group(self, group: Dict[str, Any]) -> None:
        self._check_fractional(group)
        _check(group["initial_accumulator_value"] >= 0.0, "`initial_accumulator_value` must be >= 0.")
        _check(group["epsilon"] > 0.0, "`epsilon` must be > 0.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, eps = group["lr"], group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["accumulator"] = torch.full_like(
                        p, group["initial_accumulator_value"], memory_format=torch.preserve_format
                    )
                state["step"] += 1

                grad = self._fractional_gradient(p.grad, p, group)
                acc = state["accumulator"]
                new_acc = acc + grad * grad
                acc.copy_(new_acc)
                p.sub_(lr * grad / (new_acc.sqrt() + eps))
        return loss


# =============================================================================
# FAdadelta
# =============================================================================


class FAdadelta(_FractionalHerreraBase, torch.optim.Optimizer):
    """
    Herrera-style fractional Adadelta.

        E[g^2]_t  = rho E[g^2]_{t-1} + (1 - rho) (g_t^(nu))^2
        delta_t   = - sqrt(E[dx^2]_{t-1} + eps) / sqrt(E[g^2]_t + eps) * g_t^(nu)
        E[dx^2]_t = rho E[dx^2]_{t-1} + (1 - rho) delta_t^2
        w_{t+1}   = w_t + lr * delta_t
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1.0,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        learning_rate: Optional[float] = None,
    ):
        defaults = dict(
            lr=_resolve_lr(lr, learning_rate),
            rho=float(rho),
            epsilon=float(epsilon),
            vderiv=vderiv,
            frac_epsilon=float(frac_epsilon),
        )
        super().__init__(params, defaults)

    def _validate_group(self, group: Dict[str, Any]) -> None:
        self._check_fractional(group)
        _check(0.0 <= group["rho"] < 1.0, "`rho` must satisfy 0 <= rho < 1.")
        _check(group["epsilon"] > 0.0, "`epsilon` must be > 0.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, rho, eps = group["lr"], group["rho"], group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["accumulated_grad"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["accumulated_delta"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1

                grad = self._fractional_gradient(p.grad, p, group)
                acc_g, acc_d = state["accumulated_grad"], state["accumulated_delta"]

                new_acc_g = rho * acc_g + (1.0 - rho) * grad * grad
                rms_delta = (acc_d + eps).sqrt()
                rms_grad = (new_acc_g + eps).sqrt()
                delta = -(rms_delta / rms_grad) * grad
                new_acc_d = rho * acc_d + (1.0 - rho) * delta * delta

                acc_g.copy_(new_acc_g)
                acc_d.copy_(new_acc_d)
                p.add_(lr * delta)
        return loss


# =============================================================================
# FRMSprop
# =============================================================================


class FRMSprop(_FractionalHerreraBase, torch.optim.Optimizer):
    """
    Herrera-style fractional RMSprop.

        v_t   = rho v_{t-1} + (1 - rho) (g_t^(nu))^2
        denom = sqrt(v_t + eps)                (centered: sqrt(v_t - a_t^2 + eps))
        w_{t+1} = w_t - lr * g_t^(nu) / denom   (momentum: via a momentum buffer)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.001,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        centered: bool = False,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        learning_rate: Optional[float] = None,
    ):
        defaults = dict(
            lr=_resolve_lr(lr, learning_rate),
            rho=float(rho),
            momentum=float(momentum),
            epsilon=float(epsilon),
            centered=bool(centered),
            vderiv=vderiv,
            frac_epsilon=float(frac_epsilon),
        )
        super().__init__(params, defaults)

    def _validate_group(self, group: Dict[str, Any]) -> None:
        self._check_fractional(group)
        _check(0.0 <= group["rho"] < 1.0, "`rho` must satisfy 0 <= rho < 1.")
        _check(0.0 <= group["momentum"] < 1.0, "`momentum` must satisfy 0 <= momentum < 1.")
        _check(group["epsilon"] > 0.0, "`epsilon` must be > 0.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, rho, eps, mu = group["lr"], group["rho"], group["epsilon"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["velocity"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if mu > 0.0:
                        state["momentum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["centered"]:
                        state["avg_grad"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1

                grad = self._fractional_gradient(p.grad, p, group)

                velocity = state["velocity"]
                new_velocity = rho * velocity + (1.0 - rho) * grad * grad
                velocity.copy_(new_velocity)

                if group["centered"]:
                    avg_grad = state["avg_grad"]
                    new_avg = rho * avg_grad + (1.0 - rho) * grad
                    avg_grad.copy_(new_avg)
                    denom = (new_velocity - new_avg * new_avg + eps).sqrt()
                else:
                    denom = (new_velocity + eps).sqrt()

                if mu > 0.0:
                    mom = state["momentum"]
                    new_mom = mu * mom + lr * grad / denom
                    mom.copy_(new_mom)
                    p.sub_(new_mom)
                else:
                    p.sub_(lr * grad / denom)
        return loss


# =============================================================================
# FAdamW
# =============================================================================


class FAdamW(_FractionalHerreraBase, torch.optim.Optimizer):
    """
    Herrera-style fractional AdamW. The decoupled weight decay is applied
    after the Adam step, on the updated weights (same order as the Keras and
    legacy versions):

        w' = w_t - lr * m_hat_t / (sqrt(v_hat_t) + eps)
        w_{t+1} = w' - lr * weight_decay * w'
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.001,
        weight_decay: float = 0.004,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        learning_rate: Optional[float] = None,
    ):
        defaults = dict(
            lr=_resolve_lr(lr, learning_rate),
            weight_decay=float(weight_decay),
            beta_1=float(beta_1),
            beta_2=float(beta_2),
            epsilon=float(epsilon),
            amsgrad=bool(amsgrad),
            vderiv=vderiv,
            frac_epsilon=float(frac_epsilon),
        )
        super().__init__(params, defaults)

    def _validate_group(self, group: Dict[str, Any]) -> None:
        self._check_fractional(group)
        _check(group["weight_decay"] >= 0.0, "`weight_decay` must be >= 0.")
        _check(0.0 <= group["beta_1"] < 1.0, "`beta_1` must satisfy 0 <= beta_1 < 1.")
        _check(0.0 <= group["beta_2"] < 1.0, "`beta_2` must satisfy 0 <= beta_2 < 1.")
        _check(group["epsilon"] > 0.0, "`epsilon` must be > 0.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, b1, b2, eps, wd = (group["lr"], group["beta_1"], group["beta_2"],
                                   group["epsilon"], group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        state["vhat"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]

                grad = self._fractional_gradient(p.grad, p, group)

                m, v = state["m"], state["v"]
                new_m = b1 * m + (1.0 - b1) * grad
                new_v = b2 * v + (1.0 - b2) * grad * grad
                m.copy_(new_m)
                v.copy_(new_v)

                m_hat = new_m / (1.0 - b1 ** t)
                if group["amsgrad"]:
                    vhat = state["vhat"]
                    new_vhat = torch.maximum(vhat, new_v)
                    vhat.copy_(new_vhat)
                    v_hat = new_vhat / (1.0 - b2 ** t)
                else:
                    v_hat = new_v / (1.0 - b2 ** t)

                # Adam step
                p.sub_(lr * m_hat / (v_hat.sqrt() + eps))
                # Decoupled weight decay on the updated weights
                if wd > 0.0:
                    p.sub_(lr * wd * p)
        return loss


__all__ = ["FSGD", "FAdam", "FAdagrad", "FAdadelta", "FRMSprop", "FAdamW"]


# =============================================================================
# Minimal smoke test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(128, 10)
    y = torch.randint(0, 3, (128,))

    configs = [
        ("FSGD", lambda ps: FSGD(ps, lr=1e-2, momentum=0.9, vderiv=0.9)),
        ("FAdam", lambda ps: FAdam(ps, lr=1e-3, vderiv=0.9)),
        ("FAdagrad", lambda ps: FAdagrad(ps, lr=1e-2, vderiv=0.9)),
        ("FAdadelta", lambda ps: FAdadelta(ps, lr=1.0, vderiv=0.9)),
        ("FRMSprop", lambda ps: FRMSprop(ps, lr=1e-3, vderiv=0.9)),
        ("FAdamW", lambda ps: FAdamW(ps, lr=1e-3, weight_decay=0.004, vderiv=0.9)),
    ]
    for name, make in configs:
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
        opt = make(model.parameters())
        losses = []
        for epoch in range(12):
            perm = torch.randperm(128)
            for i in range(0, 128, 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x[idx]), y[idx])
                loss.backward()
                opt.step()
            with torch.no_grad():
                losses.append(float(torch.nn.functional.cross_entropy(model(x), y)))
        print(f"  {name:10s} loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        assert losses[-1] < losses[0], f"{name}: loss did not decrease"
    print("All smoke tests passed.")
