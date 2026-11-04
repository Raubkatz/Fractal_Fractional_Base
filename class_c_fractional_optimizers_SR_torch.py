"""
class_c_fractional_optimizers_SR_torch.py

Explicit memory-based fractional (Grunwald-Letnikov) optimizers for PyTorch
(tested with PyTorch 2.15).

This file is the PyTorch counterpart of
`class_c_tf_fractional_optimizers_SR_keras3.py`: same classes, same
arguments, same update rules, written as `torch.optim.Optimizer` subclasses.
These optimizers are the GL-memory family of our article (Raubitzek et al.,
Fractal and Fractional 2026, arXiv:2608.14636), built on the classical
Grunwald-Letnikov definition (Podlubny, Fractional Differential Equations,
1999).

Definitions
-----------
Raw gradient            g_t = dL/dtheta_t
GL coefficients         c_0^(nu)=1,  c_k^(nu) = (1-(nu+1)/k) c_{k-1}^(nu)
Fractional gradient     g_t^(nu) = sum_{k=0}^{min(t,K-1)} c_k^(nu) g_{t-k}
Descent safeguard       g_t^(nu) <- g_t            if <g_t^(nu), g_t> <= 0
Norm matching           g_hat_t^(nu) = g_t^(nu) ||g_t|| / (||g_t^(nu)||+eps)
Update                  base-optimizer update with g_hat_t^(nu) as the
                        direction input; squared-gradient accumulators from
                        g_t (default) or from g_hat_t^(nu).

Switches (defaults as in the Keras version)
-------------------------------------------
norm_match=True                         rescale the GL sum to ||g_t||
descent_check=True                      fall back to g_t when the GL sum points uphill
use_raw_gradient_for_second_moment=True raw gradient in all squared-gradient accumulators
normalize_coefficients=False            divide the GL coefficients by sum |c_k|
For |nu - 1| < 1e-12 or K == 1 the history is skipped and each optimizer
is identical to its base optimizer.

The history is a circular buffer of shape [K] + param.shape per parameter:
one row is written per step (row (t-1) mod K) and the coefficients are
rotated instead of moving the data.

Implemented optimizers
----------------------
    MemoryFSGD, MemoryFRMSprop, MemoryFAdam, MemoryFAdadelta,
    MemoryFAdagrad, MemoryFAdamW, MemoryFNadam, MemoryFAdamax

PyTorch conventions (same as the other *_torch.py files)
--------------------------------------------------------
`lr=` (alias `learning_rate=`), Keras argument names for everything else,
hyperparameters per parameter group, state_dict() / load_state_dict(),
sparse gradients densified, per-parameter step counter.

Usage
-----
    from class_c_fractional_optimizers_SR_torch import MemoryFAdam

    optimizer = MemoryFAdam(model.parameters(), lr=1e-3, vderiv=0.9, history_size=6)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch


# =============================================================================
# Small helpers
# =============================================================================


def _resolve_lr(lr, learning_rate) -> float:
    value = lr if learning_rate is None else learning_rate
    if isinstance(value, torch.Tensor):
        value = float(value.item())
    return float(value)


def _dense(grad: torch.Tensor) -> torch.Tensor:
    return grad.to_dense() if grad.is_sparse else grad


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _l2(x: torch.Tensor) -> torch.Tensor:
    return (x * x).sum().sqrt()


# =============================================================================
# Shared helper base
# =============================================================================


class _MemoryFractionalBase:
    """Shared GL-memory machinery (coefficients, circular buffer, safeguards)."""

    _CLASSICAL_TOL = 1e-12

    _MEMORY_DEFAULTS = dict(
        vderiv=1.0,
        history_size=8,
        normalize_coefficients=False,
        coeff_epsilon=1e-12,
        norm_match=True,
        use_raw_gradient_for_second_moment=True,
        descent_check=True,
    )

    @classmethod
    def _memory_defaults(cls, **kwargs) -> Dict[str, Any]:
        out = dict(cls._MEMORY_DEFAULTS)
        out.update(kwargs)
        out["normalize_coefficients"] = bool(out["normalize_coefficients"])
        out["coeff_epsilon"] = float(out["coeff_epsilon"])
        out["norm_match"] = bool(out["norm_match"])
        out["use_raw_gradient_for_second_moment"] = bool(out["use_raw_gradient_for_second_moment"])
        out["descent_check"] = bool(out["descent_check"])
        return out

    @staticmethod
    def _check_memory(group: Dict[str, Any]) -> None:
        vderiv = group["vderiv"]
        if not isinstance(vderiv, (float, int)) or isinstance(vderiv, bool):
            raise TypeError("`vderiv` must be a float.")
        _check(0.0 < float(vderiv) < 2.0, "`vderiv` must satisfy 0 < vderiv < 2.")
        k = group["history_size"]
        _check(isinstance(k, int) and not isinstance(k, bool) and k >= 1,
               "`history_size` must be an integer >= 1.")
        _check(group["coeff_epsilon"] > 0.0, "`coeff_epsilon` must be > 0.")
        _check(group["lr"] >= 0.0, "`lr` must be >= 0.")

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        super().add_param_group(param_group)
        self._validate_group(self.param_groups[-1])

    # ------------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------------

    @classmethod
    def _compute_fractional_coefficients(cls, nu: float, n_terms: int,
                                         normalize: bool, coeff_epsilon: float) -> List[float]:
        """c_0 = 1, c_k = (1 - (nu + 1)/k) c_{k-1}; [1, 0, ..., 0] at nu = 1."""
        if abs(nu - 1.0) < cls._CLASSICAL_TOL:
            return [1.0] + [0.0] * (n_terms - 1)
        coeffs = [1.0]
        for k in range(1, n_terms):
            coeffs.append((1.0 - (nu + 1.0) / k) * coeffs[-1])
        if normalize:
            denom = sum(abs(c) for c in coeffs)
            if denom < coeff_epsilon:
                denom = 1.0
            coeffs = [c / denom for c in coeffs]
        return coeffs

    def _group_coefficients(self, group: Dict[str, Any]) -> List[float]:
        return self._compute_fractional_coefficients(
            float(group["vderiv"]), group["history_size"],
            group["normalize_coefficients"], group["coeff_epsilon"],
        )

    @property
    def gl_coefficients(self) -> List[float]:
        """GL coefficients c_0 ... c_{K-1} of the first parameter group."""
        return self._group_coefficients(self.param_groups[0])

    def gl_coefficient_sum(self, group_index: int = 0) -> float:
        """DC gain (sum of the GL coefficients) of a parameter group."""
        return float(sum(self._group_coefficients(self.param_groups[group_index])))

    @classmethod
    def _is_classical(cls, group: Dict[str, Any]) -> bool:
        return abs(float(group["vderiv"]) - 1.0) < cls._CLASSICAL_TOL or group["history_size"] == 1

    # ------------------------------------------------------------------
    # Fractional gradient with circular buffer
    # ------------------------------------------------------------------

    def _fractional_gradient(self, raw: torch.Tensor, p: torch.Tensor,
                             state: Dict[str, Any], group: Dict[str, Any]) -> torch.Tensor:
        """
        1. row <- (t-1) mod K, buffer[row] <- g_t           (single write)
        2. g^(nu) = sum_j c_{lag(j)} buffer[j], lag(j) = (row - j) mod K
        3. descent safeguard: g_t instead if <g^(nu), g_t> <= 0
        4. norm matching: rescale to ||g_t||
        """
        if self._is_classical(group):
            return raw

        k = group["history_size"]
        if "grad_history" not in state:
            state["grad_history"] = torch.zeros((k,) + tuple(p.shape), dtype=p.dtype, device=p.device)
        hist = state["grad_history"]

        pos = (state["step"] - 1) % k            # 0-based step index, as in Keras
        hist[pos].copy_(raw)

        coeffs = torch.tensor(self._group_coefficients(group), dtype=p.dtype, device=p.device)
        lags = torch.remainder(pos - torch.arange(k, device=p.device), k)
        c_perm = coeffs[lags]
        frac = (hist * c_perm.view((k,) + (1,) * p.dim())).sum(dim=0)

        if group["descent_check"]:
            keep = ((frac * raw).sum() > 0.0).to(p.dtype)
            frac = keep * frac + (1.0 - keep) * raw

        if group["norm_match"]:
            frac = frac * (_l2(raw) / (_l2(frac) + group["coeff_epsilon"]))

        return frac

    def _direction_and_second_moment_grads(self, p, state, group):
        raw = _dense(p.grad).to(p.dtype)
        frac = self._fractional_gradient(raw, p, state, group)
        second = raw if group["use_raw_gradient_for_second_moment"] else frac
        return raw, frac, second

    @staticmethod
    def _zeros(p):
        return torch.zeros_like(p, memory_format=torch.preserve_format)


# =============================================================================
# MemoryFSGD
# =============================================================================


class MemoryFSGD(_MemoryFractionalBase, torch.optim.Optimizer):
    """
    Explicit memory-based fractional SGD.

        theta_{t+1} = theta_t - lr * g_hat_t^(nu)
        momentum:  v_t = momentum v_{t-1} - lr g_hat_t^(nu);  theta_{t+1} = theta_t + v_t
    """

    def __init__(self, params: Iterable, lr: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False, vderiv: float = 1.0, history_size: int = 8,
                 normalize_coefficients: bool = False, coeff_epsilon: float = 1e-12,
                 norm_match: bool = True, use_raw_gradient_for_second_moment: bool = True,
                 descent_check: bool = True, learning_rate: Optional[float] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), momentum=float(momentum),
                        nesterov=bool(nesterov),
                        **self._memory_defaults(
                            vderiv=vderiv, history_size=history_size,
                            normalize_coefficients=normalize_coefficients,
                            coeff_epsilon=coeff_epsilon, norm_match=norm_match,
                            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
                            descent_check=descent_check))
        super().__init__(params, defaults)

    def _validate_group(self, group):
        self._check_memory(group)
        _check(0.0 <= group["momentum"] < 1.0, "`momentum` must satisfy 0 <= momentum < 1.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1
                _, frac, _ = self._direction_and_second_moment_grads(p, state, group)
                if mu > 0.0:
                    if "momentum" not in state:
                        state["momentum"] = self._zeros(p)
                    m = state["momentum"]
                    new_m = mu * m - lr * frac
                    m.copy_(new_m)
                    if group["nesterov"]:
                        p.add_(mu * new_m - lr * frac)
                    else:
                        p.add_(new_m)
                else:
                    p.sub_(lr * frac)
        return loss


# =============================================================================
# MemoryFRMSprop
# =============================================================================


class MemoryFRMSprop(_MemoryFractionalBase, torch.optim.Optimizer):
    """Explicit memory-based fractional RMSprop (raw-gradient accumulator by default)."""

    def __init__(self, params: Iterable, lr: float = 0.001, rho: float = 0.9,
                 momentum: float = 0.0, epsilon: float = 1e-7, centered: bool = False,
                 vderiv: float = 1.0, history_size: int = 8,
                 normalize_coefficients: bool = False, coeff_epsilon: float = 1e-12,
                 norm_match: bool = True, use_raw_gradient_for_second_moment: bool = True,
                 descent_check: bool = True, learning_rate: Optional[float] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), rho=float(rho),
                        momentum=float(momentum), epsilon=float(epsilon),
                        centered=bool(centered),
                        **self._memory_defaults(
                            vderiv=vderiv, history_size=history_size,
                            normalize_coefficients=normalize_coefficients,
                            coeff_epsilon=coeff_epsilon, norm_match=norm_match,
                            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
                            descent_check=descent_check))
        super().__init__(params, defaults)

    def _validate_group(self, group):
        self._check_memory(group)
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
                    state["velocity"] = self._zeros(p)
                    if mu > 0.0:
                        state["momentum"] = self._zeros(p)
                    if group["centered"]:
                        state["avg_grad"] = self._zeros(p)
                state["step"] += 1
                _, frac, second = self._direction_and_second_moment_grads(p, state, group)

                velocity = state["velocity"]
                new_velocity = rho * velocity + (1.0 - rho) * second * second
                velocity.copy_(new_velocity)
                if group["centered"]:
                    avg = state["avg_grad"]
                    new_avg = rho * avg + (1.0 - rho) * second
                    avg.copy_(new_avg)
                    denom = (new_velocity - new_avg * new_avg + eps).sqrt()
                else:
                    denom = (new_velocity + eps).sqrt()

                if mu > 0.0:
                    mom = state["momentum"]
                    new_mom = mu * mom + lr * frac / denom
                    mom.copy_(new_mom)
                    p.sub_(new_mom)
                else:
                    p.sub_(lr * frac / denom)
        return loss


# =============================================================================
# MemoryFAdam
# =============================================================================


class MemoryFAdam(_MemoryFractionalBase, torch.optim.Optimizer):
    """
    Explicit memory-based fractional Adam. First moment from the fractional
    gradient, second moment from the raw gradient by default. Keras-exact
    formulation: bias correction folded into the learning rate, epsilon
    outside the uncorrected sqrt.
    """

    def __init__(self, params: Iterable, lr: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, epsilon: float = 1e-7, amsgrad: bool = False,
                 vderiv: float = 1.0, history_size: int = 8,
                 normalize_coefficients: bool = False, coeff_epsilon: float = 1e-12,
                 norm_match: bool = True, use_raw_gradient_for_second_moment: bool = True,
                 descent_check: bool = True, learning_rate: Optional[float] = None,
                 _extra_defaults: Optional[Dict[str, Any]] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), beta_1=float(beta_1),
                        beta_2=float(beta_2), epsilon=float(epsilon), amsgrad=bool(amsgrad),
                        **self._memory_defaults(
                            vderiv=vderiv, history_size=history_size,
                            normalize_coefficients=normalize_coefficients,
                            coeff_epsilon=coeff_epsilon, norm_match=norm_match,
                            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
                            descent_check=descent_check),
                        **(_extra_defaults or {}))
        super().__init__(params, defaults)

    def _validate_group(self, group):
        self._check_memory(group)
        _check(0.0 <= group["beta_1"] < 1.0, "`beta_1` must satisfy 0 <= beta_1 < 1.")
        _check(0.0 <= group["beta_2"] < 1.0, "`beta_2` must satisfy 0 <= beta_2 < 1.")
        _check(group["epsilon"] > 0.0, "`epsilon` must be > 0.")

    def _adam_update(self, p, state, group):
        lr, b1, b2, eps = group["lr"], group["beta_1"], group["beta_2"], group["epsilon"]
        if "m" not in state:
            state["m"] = self._zeros(p)
            state["v"] = self._zeros(p)
            if group["amsgrad"]:
                state["vhat"] = self._zeros(p)
        t = state["step"]
        _, frac, second = self._direction_and_second_moment_grads(p, state, group)

        m, v = state["m"], state["v"]
        new_m = b1 * m + (1.0 - b1) * frac
        new_v = b2 * v + (1.0 - b2) * second * second
        m.copy_(new_m)
        v.copy_(new_v)

        lr_corr = lr * math.sqrt(1.0 - b2 ** t) / (1.0 - b1 ** t)
        if group["amsgrad"]:
            vhat = state["vhat"]
            new_vhat = torch.maximum(vhat, new_v)
            vhat.copy_(new_vhat)
            denom_v = new_vhat
        else:
            denom_v = new_v
        p.sub_(lr_corr * new_m / (denom_v.sqrt() + eps))

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1
                self._adam_update(p, state, group)
        return loss


# =============================================================================
# MemoryFAdadelta
# =============================================================================


class MemoryFAdadelta(_MemoryFractionalBase, torch.optim.Optimizer):
    """Explicit memory-based fractional Adadelta (raw-driven accumulators by default)."""

    def __init__(self, params: Iterable, lr: float = 1.0, rho: float = 0.95,
                 epsilon: float = 1e-7, vderiv: float = 1.0, history_size: int = 8,
                 normalize_coefficients: bool = False, coeff_epsilon: float = 1e-12,
                 norm_match: bool = True, use_raw_gradient_for_second_moment: bool = True,
                 descent_check: bool = True, learning_rate: Optional[float] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), rho=float(rho), epsilon=float(epsilon),
                        **self._memory_defaults(
                            vderiv=vderiv, history_size=history_size,
                            normalize_coefficients=normalize_coefficients,
                            coeff_epsilon=coeff_epsilon, norm_match=norm_match,
                            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
                            descent_check=descent_check))
        super().__init__(params, defaults)

    def _validate_group(self, group):
        self._check_memory(group)
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
                    state["accumulated_grad"] = self._zeros(p)
                    state["accumulated_delta"] = self._zeros(p)
                state["step"] += 1
                _, frac, second = self._direction_and_second_moment_grads(p, state, group)

                acc_g, acc_d = state["accumulated_grad"], state["accumulated_delta"]
                new_acc_g = rho * acc_g + (1.0 - rho) * second * second
                rms_delta = (acc_d + eps).sqrt()
                rms_grad = (new_acc_g + eps).sqrt()
                delta = -(rms_delta / rms_grad) * frac
                new_acc_d = rho * acc_d + (1.0 - rho) * delta * delta
                acc_g.copy_(new_acc_g)
                acc_d.copy_(new_acc_d)
                p.add_(lr * delta)
        return loss


# =============================================================================
# MemoryFAdagrad
# =============================================================================


class MemoryFAdagrad(_MemoryFractionalBase, torch.optim.Optimizer):
    """
    Explicit memory-based fractional Adagrad.

        acc_t = acc_{t-1} + s_t^2   (s_t = raw gradient by default)
        theta_{t+1} = theta_t - lr * g_hat_t^(nu) / (sqrt(acc_t) + eps)
    """

    def __init__(self, params: Iterable, lr: float = 0.001,
                 initial_accumulator_value: float = 0.1, epsilon: float = 1e-7,
                 vderiv: float = 1.0, history_size: int = 8,
                 normalize_coefficients: bool = False, coeff_epsilon: float = 1e-12,
                 norm_match: bool = True, use_raw_gradient_for_second_moment: bool = True,
                 descent_check: bool = True, learning_rate: Optional[float] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate),
                        initial_accumulator_value=float(initial_accumulator_value),
                        epsilon=float(epsilon),
                        **self._memory_defaults(
                            vderiv=vderiv, history_size=history_size,
                            normalize_coefficients=normalize_coefficients,
                            coeff_epsilon=coeff_epsilon, norm_match=norm_match,
                            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
                            descent_check=descent_check))
        super().__init__(params, defaults)

    def _validate_group(self, group):
        self._check_memory(group)
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
                        p, group["initial_accumulator_value"], memory_format=torch.preserve_format)
                state["step"] += 1
                _, frac, second = self._direction_and_second_moment_grads(p, state, group)
                acc = state["accumulator"]
                new_acc = acc + second * second
                acc.copy_(new_acc)
                p.sub_(lr * frac / (new_acc.sqrt() + eps))
        return loss


# =============================================================================
# MemoryFAdamW
# =============================================================================


class MemoryFAdamW(MemoryFAdam):
    """
    MemoryFAdam plus decoupled weight decay after the Adam step, on the
    updated weights:  theta_{t+1} = theta' - lr * weight_decay * theta'.
    """

    def __init__(self, params: Iterable, lr: float = 0.001, weight_decay: float = 0.004,
                 learning_rate: Optional[float] = None, **kwargs):
        if weight_decay is None or weight_decay < 0.0:
            raise ValueError("`weight_decay` must be >= 0.")
        super().__init__(params, lr=lr, learning_rate=learning_rate,
                         _extra_defaults={"weight_decay": float(weight_decay)}, **kwargs)

    def _validate_group(self, group):
        super()._validate_group(group)
        _check(group["weight_decay"] >= 0.0, "`weight_decay` must be >= 0.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1
                self._adam_update(p, state, group)
                if wd > 0.0:
                    p.sub_(lr * wd * p)
        return loss


# =============================================================================
# MemoryFNadam
# =============================================================================


class MemoryFNadam(_MemoryFractionalBase, torch.optim.Optimizer):
    """
    Explicit memory-based fractional Nadam (Dozat's formulation without the
    momentum-product schedule):
        m_bar = beta_1 m_t / (1 - beta_1^{t+1}) + (1 - beta_1) g_hat_t / (1 - beta_1^t)
        theta_{t+1} = theta_t - lr * m_bar / (sqrt(v_t / (1 - beta_2^t)) + eps)
    """

    def __init__(self, params: Iterable, lr: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, epsilon: float = 1e-7, vderiv: float = 1.0,
                 history_size: int = 8, normalize_coefficients: bool = False,
                 coeff_epsilon: float = 1e-12, norm_match: bool = True,
                 use_raw_gradient_for_second_moment: bool = True, descent_check: bool = True,
                 learning_rate: Optional[float] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), beta_1=float(beta_1),
                        beta_2=float(beta_2), epsilon=float(epsilon),
                        **self._memory_defaults(
                            vderiv=vderiv, history_size=history_size,
                            normalize_coefficients=normalize_coefficients,
                            coeff_epsilon=coeff_epsilon, norm_match=norm_match,
                            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
                            descent_check=descent_check))
        super().__init__(params, defaults)

    def _validate_group(self, group):
        self._check_memory(group)
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
                    state["m"] = self._zeros(p)
                    state["v"] = self._zeros(p)
                state["step"] += 1
                t = state["step"]
                _, frac, second = self._direction_and_second_moment_grads(p, state, group)

                m, v = state["m"], state["v"]
                new_m = b1 * m + (1.0 - b1) * frac
                new_v = b2 * v + (1.0 - b2) * second * second
                m.copy_(new_m)
                v.copy_(new_v)

                b1_pow = b1 ** t
                m_bar = b1 * new_m / (1.0 - b1_pow * b1) + (1.0 - b1) * frac / (1.0 - b1_pow)
                v_hat = new_v / (1.0 - b2 ** t)
                p.sub_(lr * m_bar / (v_hat.sqrt() + eps))
        return loss


# =============================================================================
# MemoryFAdamax
# =============================================================================


class MemoryFAdamax(_MemoryFractionalBase, torch.optim.Optimizer):
    """
    Explicit memory-based fractional Adamax:
        u_t = max(beta_2 u_{t-1}, |s_t|)
        theta_{t+1} = theta_t - lr / (1 - beta_1^t) * m_t / (u_t + eps)
    """

    def __init__(self, params: Iterable, lr: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, epsilon: float = 1e-7, vderiv: float = 1.0,
                 history_size: int = 8, normalize_coefficients: bool = False,
                 coeff_epsilon: float = 1e-12, norm_match: bool = True,
                 use_raw_gradient_for_second_moment: bool = True, descent_check: bool = True,
                 learning_rate: Optional[float] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), beta_1=float(beta_1),
                        beta_2=float(beta_2), epsilon=float(epsilon),
                        **self._memory_defaults(
                            vderiv=vderiv, history_size=history_size,
                            normalize_coefficients=normalize_coefficients,
                            coeff_epsilon=coeff_epsilon, norm_match=norm_match,
                            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
                            descent_check=descent_check))
        super().__init__(params, defaults)

    def _validate_group(self, group):
        self._check_memory(group)
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
                    state["m"] = self._zeros(p)
                    state["u"] = self._zeros(p)
                state["step"] += 1
                t = state["step"]
                _, frac, second = self._direction_and_second_moment_grads(p, state, group)

                m, u = state["m"], state["u"]
                new_m = b1 * m + (1.0 - b1) * frac
                new_u = torch.maximum(b2 * u, second.abs())
                m.copy_(new_m)
                u.copy_(new_u)
                p.sub_(lr / (1.0 - b1 ** t) * new_m / (new_u + eps))
        return loss


__all__ = [
    "MemoryFSGD", "MemoryFRMSprop", "MemoryFAdam", "MemoryFAdadelta",
    "MemoryFAdagrad", "MemoryFAdamW", "MemoryFNadam", "MemoryFAdamax",
]


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(128, 10)
    y = torch.randint(0, 3, (128,))
    configs = [
        ("MemoryFSGD nu=0.9", lambda ps: MemoryFSGD(ps, lr=1e-2, vderiv=0.9, history_size=6)),
        ("MemoryFSGD nu=1.5", lambda ps: MemoryFSGD(ps, lr=1e-2, vderiv=1.5, history_size=6)),
        ("MemoryFRMSprop", lambda ps: MemoryFRMSprop(ps, lr=1e-3, vderiv=0.9, history_size=6)),
        ("MemoryFAdam", lambda ps: MemoryFAdam(ps, lr=1e-3, vderiv=0.9, history_size=6)),
        ("MemoryFAdadelta", lambda ps: MemoryFAdadelta(ps, lr=1.0, vderiv=0.9, history_size=6)),
        ("MemoryFAdagrad", lambda ps: MemoryFAdagrad(ps, lr=1e-2, vderiv=0.9, history_size=6)),
        ("MemoryFAdamW", lambda ps: MemoryFAdamW(ps, lr=1e-3, weight_decay=0.004, vderiv=0.9, history_size=6)),
        ("MemoryFNadam", lambda ps: MemoryFNadam(ps, lr=1e-3, vderiv=0.9, history_size=6)),
        ("MemoryFAdamax", lambda ps: MemoryFAdamax(ps, lr=1e-3, vderiv=0.9, history_size=6)),
        ("MemoryFAdam nu=1", lambda ps: MemoryFAdam(ps, lr=1e-3, vderiv=1.0, history_size=6)),
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
                torch.nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
                opt.step()
            with torch.no_grad():
                losses.append(float(torch.nn.functional.cross_entropy(model(x), y)))
        print(f"  {name:22s} loss {losses[0]:.4f} -> {losses[-1]:.4f}  DC gain {opt.gl_coefficient_sum():+.4f}")
        assert losses[-1] < losses[0], f"{name}: loss did not decrease"
    print("All smoke tests passed.")
