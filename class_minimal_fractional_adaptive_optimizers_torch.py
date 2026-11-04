"""
class_minimal_fractional_adaptive_optimizers_torch.py

Adaptive memory-based fractional optimizers for PyTorch (tested with
PyTorch 2.15).

This file is the PyTorch counterpart of
`class_minimal_fractional_adaptive_optimizers_keras3.py`: same classes, same
arguments, same update rules, written as `torch.optim.Optimizer` subclasses.
These optimizers were introduced in our article (Raubitzek et al., Fractal
and Fractional 2026, arXiv:2608.14636).

Main idea
---------
    1. Keep the fractional order nu fixed.
    2. Short-memory GL gradient g_t^(nu) = sum_{k=0}^{K-1} c_k^(nu) g_{t-k},
       c_0 = 1, c_k = (1 - (nu+1)/k) c_{k-1}.
    3. Descent safeguard (g_t if <g_t^(nu), g_t> <= 0), then norm matching
       g_hat_t^(nu) = g_t^(nu) ||g_t|| / (||g_t^(nu)|| + eps).
    4. Mix:  g_eff = (1 - lambda_t) g_t + lambda_t g_hat_t^(nu).
    5. Adapt only lambda_t:
           s_t      = min(||g_t - g_{t-1}|| / (||g_{t-1}|| + eps), clip)
           S_t      = gamma_s S_{t-1} + (1 - gamma_s) s_t          (stability EMA)
           target   = lambda_{t-1} + step  if S_t <= threshold, else  - step
           target  -= loss_mix_decrease    if the smoothed loss got worse
           lambda_t = clip(gamma_m lambda_{t-1} + (1 - gamma_m) clip(target), min_mix, max_mix_t)
lambda_t = 0 gives exactly the base optimizer; at nu = 1, g_eff = g_t.

Implemented optimizers
----------------------
    AdaptiveMemoryFSGD, AdaptiveMemoryFRMSprop, AdaptiveMemoryFAdam,
    AdaptiveMemoryFAdadelta, AdaptiveMemoryFAdagrad, AdaptiveMemoryFAdamW,
    AdaptiveMemoryFNadam, AdaptiveMemoryFAdamax

Recommended defaults (unchanged)
--------------------------------
Neural networks:  vderiv = 0.85-0.9, history_size = 3-4, initial_mix = 0.10,
                  max_mix = 0.30, adapt_step = 0.01, stability_threshold = 0.25.

Optional loss feedback
----------------------
Call `optimizer.set_current_loss(loss)` after each step of your training
loop to lower lambda_t whenever the smoothed loss gets worse than the best
smoothed loss so far. Not calling it (the paper setting) leaves the loss term
at zero. The loss-feedback state is included in `state_dict()`.
`loss_ema_beta` and `loss_worsen_tolerance` are read from the optimizer
defaults (one loss per optimizer); every other setting can differ per
parameter group.

PyTorch conventions (same as the other *_torch.py files)
--------------------------------------------------------
`lr=` (alias `learning_rate=`), Keras argument names for everything else,
hyperparameters per parameter group, state_dict() / load_state_dict(),
sparse gradients densified, per-parameter step counter.

Usage
-----
    from class_minimal_fractional_adaptive_optimizers_torch import AdaptiveMemoryFAdam

    optimizer = AdaptiveMemoryFAdam(model.parameters(), lr=1e-3, vderiv=0.9, history_size=4)
    ...
    loss.backward(); optimizer.step()
    optimizer.set_current_loss(loss)            # optional
    print(optimizer.current_mix_values())       # lambda_t per parameter
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


_ADAPTIVE_DEFAULTS = dict(
    vderiv=0.9,
    history_size=4,
    normalize_coefficients=True,
    coeff_epsilon=1e-12,
    initial_mix=0.10,
    min_mix=0.0,
    max_mix=0.30,
    adapt_step=0.01,
    mix_ema_gamma=0.90,
    stability_ema_gamma=0.95,
    stability_threshold=0.25,
    stability_clip=10.0,
    norm_match_fractional_gradient=True,
    use_raw_gradient_for_second_moment=True,
    loss_ema_beta=0.95,
    loss_worsen_tolerance=1e-4,
    loss_mix_decrease=0.02,
    warmup_steps=0,
    descent_check=True,
)


def _split_adaptive_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pop the adaptive-memory keywords; anything left over is an error."""
    adaptive = dict(_ADAPTIVE_DEFAULTS)
    for key in list(kwargs):
        if key in _ADAPTIVE_DEFAULTS:
            adaptive[key] = kwargs.pop(key)
    if kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {sorted(kwargs)}")
    for key in ("normalize_coefficients", "norm_match_fractional_gradient",
                "use_raw_gradient_for_second_moment", "descent_check"):
        adaptive[key] = bool(adaptive[key])
    for key in ("coeff_epsilon", "initial_mix", "min_mix", "max_mix", "adapt_step",
                "mix_ema_gamma", "stability_ema_gamma", "stability_threshold",
                "stability_clip", "loss_ema_beta", "loss_worsen_tolerance",
                "loss_mix_decrease"):
        adaptive[key] = float(adaptive[key])
    if isinstance(adaptive["warmup_steps"], float) and adaptive["warmup_steps"].is_integer():
        adaptive["warmup_steps"] = int(adaptive["warmup_steps"])
    return adaptive


# =============================================================================
# Shared adaptive memory-fractional helper base
# =============================================================================


class _AdaptiveMemoryFractionalBase:
    """Shared machinery: GL coefficients, circular buffer, adaptive lambda_t, loss feedback."""

    _CLASSICAL_TOL = 1e-12

    def _init_loss_feedback(self) -> None:
        self._loss_feedback = {"current": None, "ema": None, "best": None, "worse": 0.0}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _check_adaptive(g: Dict[str, Any]) -> None:
        vderiv = g["vderiv"]
        if not isinstance(vderiv, (float, int)) or isinstance(vderiv, bool):
            raise TypeError("`vderiv` must be a float.")
        _check(0.0 < float(vderiv) < 2.0, "`vderiv` must satisfy 0 < vderiv < 2.")
        k = g["history_size"]
        _check(isinstance(k, int) and not isinstance(k, bool) and k >= 1,
               "`history_size` must be an integer >= 1.")
        _check(g["coeff_epsilon"] > 0.0, "`coeff_epsilon` must be > 0.")
        _check(0.0 <= g["min_mix"] <= g["initial_mix"] <= g["max_mix"] <= 1.0,
               "Mix values must satisfy 0 <= min_mix <= initial_mix <= max_mix <= 1.")
        _check(g["adapt_step"] >= 0.0, "`adapt_step` must be >= 0.")
        _check(0.0 <= g["mix_ema_gamma"] < 1.0, "`mix_ema_gamma` must satisfy 0 <= mix_ema_gamma < 1.")
        _check(0.0 <= g["stability_ema_gamma"] < 1.0,
               "`stability_ema_gamma` must satisfy 0 <= stability_ema_gamma < 1.")
        _check(g["stability_threshold"] >= 0.0, "`stability_threshold` must be >= 0.")
        _check(g["stability_clip"] > 0.0, "`stability_clip` must be > 0.")
        _check(0.0 <= g["loss_ema_beta"] < 1.0, "`loss_ema_beta` must satisfy 0 <= loss_ema_beta < 1.")
        _check(g["loss_worsen_tolerance"] >= 0.0, "`loss_worsen_tolerance` must be >= 0.")
        _check(g["loss_mix_decrease"] >= 0.0, "`loss_mix_decrease` must be >= 0.")
        _check(isinstance(g["warmup_steps"], int) and g["warmup_steps"] >= 0,
               "`warmup_steps` must be an integer >= 0.")
        _check(g["lr"] >= 0.0, "`lr` must be >= 0.")

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        super().add_param_group(param_group)
        self._validate_group(self.param_groups[-1])

    # ------------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------------

    @classmethod
    def _compute_fractional_coefficients(cls, nu, n_terms, normalize, coeff_epsilon) -> List[float]:
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

    def _group_coefficients(self, g) -> List[float]:
        return self._compute_fractional_coefficients(
            float(g["vderiv"]), g["history_size"], g["normalize_coefficients"], g["coeff_epsilon"])

    @property
    def gl_coefficients(self) -> List[float]:
        """GL coefficients of the first parameter group."""
        return self._group_coefficients(self.param_groups[0])

    def gl_coefficient_sum(self, group_index: int = 0) -> float:
        """DC gain of the (possibly normalized) GL kernel of a parameter group."""
        return float(sum(self._group_coefficients(self.param_groups[group_index])))

    @classmethod
    def _is_classical(cls, g) -> bool:
        return abs(float(g["vderiv"]) - 1.0) < cls._CLASSICAL_TOL or g["history_size"] == 1

    # ------------------------------------------------------------------
    # Loss feedback
    # ------------------------------------------------------------------

    def set_current_loss(self, loss_value) -> None:
        """
        Optional loss injection from the training loop. When the smoothed
        loss becomes worse than the best smoothed loss so far, lambda_t is
        pushed down by `loss_mix_decrease` in the following steps.
        """
        # rounded to float32 first, exactly like the Keras version
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.detach().cpu()
        loss_scalar = float(torch.as_tensor(loss_value, dtype=torch.float32).item())
        beta = float(self.defaults["loss_ema_beta"])
        tol = float(self.defaults["loss_worsen_tolerance"])
        lf = self._loss_feedback
        lf["current"] = loss_scalar

        if lf["ema"] is None:
            lf["ema"] = loss_scalar
            lf["best"] = loss_scalar
            lf["worse"] = 0.0
            return

        lf["ema"] = beta * lf["ema"] + (1.0 - beta) * loss_scalar
        threshold = tol * max(1.0, abs(lf["best"]))
        if lf["ema"] < lf["best"] - threshold:
            lf["best"] = lf["ema"]
            lf["worse"] = 0.0
        elif lf["ema"] > lf["best"] + threshold:
            lf["worse"] = 1.0
        else:
            lf["worse"] = 0.0

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd["loss_feedback"] = dict(self._loss_feedback)
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        lf = state_dict.pop("loss_feedback", None)
        super().load_state_dict(state_dict)
        if lf is not None:
            self._loss_feedback = dict(lf)

    # ------------------------------------------------------------------
    # Fractional gradient and adaptive mix
    # ------------------------------------------------------------------

    def _fractional_gradient(self, raw, p, state, g):
        if self._is_classical(g):
            return raw
        k = g["history_size"]
        hist = state["grad_history"]
        pos = (state["step"] - 1) % k
        hist[pos].copy_(raw)

        coeffs = torch.tensor(self._group_coefficients(g), dtype=p.dtype, device=p.device)
        lags = torch.remainder(pos - torch.arange(k, device=p.device), k)
        frac = (hist * coeffs[lags].view((k,) + (1,) * p.dim())).sum(dim=0)

        if g["descent_check"]:
            keep = ((frac * raw).sum() > 0.0).to(p.dtype)
            frac = keep * frac + (1.0 - keep) * raw
        if g["norm_match_fractional_gradient"]:
            frac = frac * (_l2(raw) / (_l2(frac) + g["coeff_epsilon"]))
        return frac

    def _init_adaptive_state(self, p, state, g) -> None:
        state["step"] = 0
        if not self._is_classical(g):
            state["grad_history"] = torch.zeros((g["history_size"],) + tuple(p.shape),
                                                dtype=p.dtype, device=p.device)
        state["prev_raw_grad"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["frac_mix"] = torch.full((), g["initial_mix"], dtype=p.dtype, device=p.device)
        state["stability_state"] = torch.zeros((), dtype=p.dtype, device=p.device)

    def _adaptive_effective_gradient(self, p, state, g):
        """Returns (raw_grad, eff_grad, second_moment_grad); updates the adaptive state."""
        raw = _dense(p.grad).to(p.dtype)
        prev = state["prev_raw_grad"]
        mix = state["frac_mix"]
        stab = state["stability_state"]

        frac = self._fractional_gradient(raw, p, state, g)

        score = _l2(raw - prev) / (_l2(prev) + g["coeff_epsilon"])
        score = torch.clamp(score, max=g["stability_clip"])

        gamma_s = g["stability_ema_gamma"]
        new_stab = gamma_s * stab + (1.0 - gamma_s) * score
        stable = new_stab <= g["stability_threshold"]

        step = g["adapt_step"]
        target = torch.where(stable, mix + step, mix - step)
        target = target - self._loss_feedback["worse"] * g["loss_mix_decrease"]

        if g["warmup_steps"] > 0:
            fraction = min(max(state["step"] / g["warmup_steps"], 0.0), 1.0)
            max_mix_t = g["max_mix"] * fraction
        else:
            max_mix_t = g["max_mix"]
        min_mix_t = g["min_mix"]
        target = torch.clamp(target, min_mix_t, max_mix_t)

        gamma_m = g["mix_ema_gamma"]
        new_mix = gamma_m * mix + (1.0 - gamma_m) * target
        new_mix = torch.clamp(new_mix, min_mix_t, max_mix_t)

        eff = (1.0 - new_mix) * raw + new_mix * frac
        second = raw if g["use_raw_gradient_for_second_moment"] else eff

        prev.copy_(raw)
        mix.copy_(new_mix)
        stab.copy_(new_stab)
        return raw, eff, second

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def _states_in_order(self, key: str) -> List[torch.Tensor]:
        out = []
        for g in self.param_groups:
            for p in g["params"]:
                st = self.state.get(p, {})
                if key in st:
                    out.append(st[key].detach().clone())
        return out

    def current_mix_values(self) -> List[torch.Tensor]:
        """lambda_t, one scalar per parameter (parameter-group order)."""
        return self._states_in_order("frac_mix")

    def current_stability_values(self) -> List[torch.Tensor]:
        """Smoothed stability score S_t, one scalar per parameter."""
        return self._states_in_order("stability_state")

    @staticmethod
    def _zeros(p):
        return torch.zeros_like(p, memory_format=torch.preserve_format)

    def _run(self, closure, update_fn):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    self._init_adaptive_state(p, state, g)
                state["step"] += 1
                update_fn(p, state, g)
        return loss


# =============================================================================
# AdaptiveMemoryFSGD
# =============================================================================


class AdaptiveMemoryFSGD(_AdaptiveMemoryFractionalBase, torch.optim.Optimizer):
    """Adaptive memory-fractional SGD (optional momentum / Nesterov)."""

    def __init__(self, params: Iterable, lr: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False, learning_rate: Optional[float] = None, **kwargs):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), momentum=float(momentum),
                        nesterov=bool(nesterov), **_split_adaptive_kwargs(kwargs))
        self._init_loss_feedback()
        super().__init__(params, defaults)

    def _validate_group(self, g):
        self._check_adaptive(g)
        _check(0.0 <= g["momentum"] < 1.0, "`momentum` must satisfy 0 <= momentum < 1.")

    def _update(self, p, state, g):
        lr, mu = g["lr"], g["momentum"]
        _, eff, _ = self._adaptive_effective_gradient(p, state, g)
        if mu > 0.0:
            if "momentum" not in state:
                state["momentum"] = self._zeros(p)
            m = state["momentum"]
            new_m = mu * m - lr * eff
            m.copy_(new_m)
            if g["nesterov"]:
                p.add_(mu * new_m - lr * eff)
            else:
                p.add_(new_m)
        else:
            p.sub_(lr * eff)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        return self._run(closure, self._update)


# =============================================================================
# AdaptiveMemoryFRMSprop
# =============================================================================


class AdaptiveMemoryFRMSprop(_AdaptiveMemoryFractionalBase, torch.optim.Optimizer):
    """Adaptive memory-fractional RMSprop (raw-gradient accumulator by default)."""

    def __init__(self, params: Iterable, lr: float = 0.001, rho: float = 0.9,
                 momentum: float = 0.0, epsilon: float = 1e-7, centered: bool = False,
                 learning_rate: Optional[float] = None, **kwargs):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), rho=float(rho), momentum=float(momentum),
                        epsilon=float(epsilon), centered=bool(centered),
                        **_split_adaptive_kwargs(kwargs))
        self._init_loss_feedback()
        super().__init__(params, defaults)

    def _validate_group(self, g):
        self._check_adaptive(g)
        _check(0.0 <= g["rho"] < 1.0, "`rho` must satisfy 0 <= rho < 1.")
        _check(0.0 <= g["momentum"] < 1.0, "`momentum` must satisfy 0 <= momentum < 1.")
        _check(g["epsilon"] > 0.0, "`epsilon` must be > 0.")

    def _update(self, p, state, g):
        lr, rho, eps, mu = g["lr"], g["rho"], g["epsilon"], g["momentum"]
        if "velocity" not in state:
            state["velocity"] = self._zeros(p)
            if mu > 0.0:
                state["momentum"] = self._zeros(p)
            if g["centered"]:
                state["avg_grad"] = self._zeros(p)
        _, eff, second = self._adaptive_effective_gradient(p, state, g)

        velocity = state["velocity"]
        new_velocity = rho * velocity + (1.0 - rho) * second * second
        velocity.copy_(new_velocity)
        if g["centered"]:
            avg = state["avg_grad"]
            new_avg = rho * avg + (1.0 - rho) * second
            avg.copy_(new_avg)
            denom = (new_velocity - new_avg * new_avg + eps).sqrt()
        else:
            denom = (new_velocity + eps).sqrt()
        if mu > 0.0:
            mom = state["momentum"]
            new_mom = mu * mom + lr * eff / denom
            mom.copy_(new_mom)
            p.sub_(new_mom)
        else:
            p.sub_(lr * eff / denom)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        return self._run(closure, self._update)


# =============================================================================
# AdaptiveMemoryFAdam
# =============================================================================


class AdaptiveMemoryFAdam(_AdaptiveMemoryFractionalBase, torch.optim.Optimizer):
    """
    Adaptive memory-fractional Adam. Mixed gradient in the first moment, raw
    gradient in the second moment (default). Keras-exact Adam formulation.
    """

    def __init__(self, params: Iterable, lr: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, epsilon: float = 1e-7, amsgrad: bool = False,
                 learning_rate: Optional[float] = None,
                 _extra_defaults: Optional[Dict[str, Any]] = None, **kwargs):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), beta_1=float(beta_1),
                        beta_2=float(beta_2), epsilon=float(epsilon), amsgrad=bool(amsgrad),
                        **_split_adaptive_kwargs(kwargs), **(_extra_defaults or {}))
        self._init_loss_feedback()
        super().__init__(params, defaults)

    def _validate_group(self, g):
        self._check_adaptive(g)
        _check(0.0 <= g["beta_1"] < 1.0, "`beta_1` must satisfy 0 <= beta_1 < 1.")
        _check(0.0 <= g["beta_2"] < 1.0, "`beta_2` must satisfy 0 <= beta_2 < 1.")
        _check(g["epsilon"] > 0.0, "`epsilon` must be > 0.")

    def _update(self, p, state, g):
        lr, b1, b2, eps = g["lr"], g["beta_1"], g["beta_2"], g["epsilon"]
        if "m" not in state:
            state["m"] = self._zeros(p)
            state["v"] = self._zeros(p)
            if g["amsgrad"]:
                state["vhat"] = self._zeros(p)
        t = state["step"]
        _, eff, second = self._adaptive_effective_gradient(p, state, g)

        m, v = state["m"], state["v"]
        new_m = b1 * m + (1.0 - b1) * eff
        new_v = b2 * v + (1.0 - b2) * second * second
        m.copy_(new_m)
        v.copy_(new_v)

        lr_corr = lr * math.sqrt(1.0 - b2 ** t) / (1.0 - b1 ** t)
        if g["amsgrad"]:
            vhat = state["vhat"]
            new_vhat = torch.maximum(vhat, new_v)
            vhat.copy_(new_vhat)
            denom_v = new_vhat
        else:
            denom_v = new_v
        p.sub_(lr_corr * new_m / (denom_v.sqrt() + eps))

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        return self._run(closure, self._update)


# =============================================================================
# AdaptiveMemoryFAdadelta
# =============================================================================


class AdaptiveMemoryFAdadelta(_AdaptiveMemoryFractionalBase, torch.optim.Optimizer):
    """Adaptive memory-fractional Adadelta."""

    def __init__(self, params: Iterable, lr: float = 1.0, rho: float = 0.95,
                 epsilon: float = 1e-7, learning_rate: Optional[float] = None, **kwargs):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), rho=float(rho), epsilon=float(epsilon),
                        **_split_adaptive_kwargs(kwargs))
        self._init_loss_feedback()
        super().__init__(params, defaults)

    def _validate_group(self, g):
        self._check_adaptive(g)
        _check(0.0 <= g["rho"] < 1.0, "`rho` must satisfy 0 <= rho < 1.")
        _check(g["epsilon"] > 0.0, "`epsilon` must be > 0.")

    def _update(self, p, state, g):
        lr, rho, eps = g["lr"], g["rho"], g["epsilon"]
        if "accumulated_grad" not in state:
            state["accumulated_grad"] = self._zeros(p)
            state["accumulated_delta"] = self._zeros(p)
        _, eff, second = self._adaptive_effective_gradient(p, state, g)

        acc_g, acc_d = state["accumulated_grad"], state["accumulated_delta"]
        new_acc_g = rho * acc_g + (1.0 - rho) * second * second
        rms_delta = (acc_d + eps).sqrt()
        rms_grad = (new_acc_g + eps).sqrt()
        delta = -(rms_delta / rms_grad) * eff
        new_acc_d = rho * acc_d + (1.0 - rho) * delta * delta
        acc_g.copy_(new_acc_g)
        acc_d.copy_(new_acc_d)
        p.add_(lr * delta)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        return self._run(closure, self._update)


# =============================================================================
# AdaptiveMemoryFAdagrad
# =============================================================================


class AdaptiveMemoryFAdagrad(_AdaptiveMemoryFractionalBase, torch.optim.Optimizer):
    """Adaptive memory-fractional Adagrad."""

    def __init__(self, params: Iterable, lr: float = 0.001,
                 initial_accumulator_value: float = 0.1, epsilon: float = 1e-7,
                 learning_rate: Optional[float] = None, **kwargs):
        defaults = dict(lr=_resolve_lr(lr, learning_rate),
                        initial_accumulator_value=float(initial_accumulator_value),
                        epsilon=float(epsilon), **_split_adaptive_kwargs(kwargs))
        self._init_loss_feedback()
        super().__init__(params, defaults)

    def _validate_group(self, g):
        self._check_adaptive(g)
        _check(g["initial_accumulator_value"] >= 0.0, "`initial_accumulator_value` must be >= 0.")
        _check(g["epsilon"] > 0.0, "`epsilon` must be > 0.")

    def _update(self, p, state, g):
        lr, eps = g["lr"], g["epsilon"]
        if "accumulator" not in state:
            state["accumulator"] = torch.full_like(p, g["initial_accumulator_value"],
                                                   memory_format=torch.preserve_format)
        _, eff, second = self._adaptive_effective_gradient(p, state, g)
        acc = state["accumulator"]
        new_acc = acc + second * second
        acc.copy_(new_acc)
        p.sub_(lr * eff / (new_acc.sqrt() + eps))

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        return self._run(closure, self._update)


# =============================================================================
# AdaptiveMemoryFAdamW
# =============================================================================


class AdaptiveMemoryFAdamW(AdaptiveMemoryFAdam):
    """AdaptiveMemoryFAdam plus decoupled weight decay after the Adam step."""

    def __init__(self, params: Iterable, lr: float = 0.001, weight_decay: float = 0.004,
                 learning_rate: Optional[float] = None, **kwargs):
        if weight_decay is None or weight_decay < 0.0:
            raise ValueError("`weight_decay` must be >= 0.")
        super().__init__(params, lr=lr, learning_rate=learning_rate,
                         _extra_defaults={"weight_decay": float(weight_decay)}, **kwargs)

    def _validate_group(self, g):
        super()._validate_group(g)
        _check(g["weight_decay"] >= 0.0, "`weight_decay` must be >= 0.")

    def _update(self, p, state, g):
        super()._update(p, state, g)
        if g["weight_decay"] > 0.0:
            p.sub_(g["lr"] * g["weight_decay"] * p)


# =============================================================================
# AdaptiveMemoryFNadam
# =============================================================================


class AdaptiveMemoryFNadam(_AdaptiveMemoryFractionalBase, torch.optim.Optimizer):
    """Adaptive memory-fractional Nadam (Dozat's formulation)."""

    def __init__(self, params: Iterable, lr: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, epsilon: float = 1e-7,
                 learning_rate: Optional[float] = None, **kwargs):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), beta_1=float(beta_1),
                        beta_2=float(beta_2), epsilon=float(epsilon),
                        **_split_adaptive_kwargs(kwargs))
        self._init_loss_feedback()
        super().__init__(params, defaults)

    def _validate_group(self, g):
        self._check_adaptive(g)
        _check(0.0 <= g["beta_1"] < 1.0, "`beta_1` must satisfy 0 <= beta_1 < 1.")
        _check(0.0 <= g["beta_2"] < 1.0, "`beta_2` must satisfy 0 <= beta_2 < 1.")
        _check(g["epsilon"] > 0.0, "`epsilon` must be > 0.")

    def _update(self, p, state, g):
        lr, b1, b2, eps = g["lr"], g["beta_1"], g["beta_2"], g["epsilon"]
        if "m" not in state:
            state["m"] = self._zeros(p)
            state["v"] = self._zeros(p)
        t = state["step"]
        _, eff, second = self._adaptive_effective_gradient(p, state, g)
        m, v = state["m"], state["v"]
        new_m = b1 * m + (1.0 - b1) * eff
        new_v = b2 * v + (1.0 - b2) * second * second
        m.copy_(new_m)
        v.copy_(new_v)
        b1_pow = b1 ** t
        m_bar = b1 * new_m / (1.0 - b1_pow * b1) + (1.0 - b1) * eff / (1.0 - b1_pow)
        v_hat = new_v / (1.0 - b2 ** t)
        p.sub_(lr * m_bar / (v_hat.sqrt() + eps))

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        return self._run(closure, self._update)


# =============================================================================
# AdaptiveMemoryFAdamax
# =============================================================================


class AdaptiveMemoryFAdamax(_AdaptiveMemoryFractionalBase, torch.optim.Optimizer):
    """Adaptive memory-fractional Adamax (infinity-norm second moment)."""

    def __init__(self, params: Iterable, lr: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, epsilon: float = 1e-7,
                 learning_rate: Optional[float] = None, **kwargs):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), beta_1=float(beta_1),
                        beta_2=float(beta_2), epsilon=float(epsilon),
                        **_split_adaptive_kwargs(kwargs))
        self._init_loss_feedback()
        super().__init__(params, defaults)

    def _validate_group(self, g):
        self._check_adaptive(g)
        _check(0.0 <= g["beta_1"] < 1.0, "`beta_1` must satisfy 0 <= beta_1 < 1.")
        _check(0.0 <= g["beta_2"] < 1.0, "`beta_2` must satisfy 0 <= beta_2 < 1.")
        _check(g["epsilon"] > 0.0, "`epsilon` must be > 0.")

    def _update(self, p, state, g):
        lr, b1, b2, eps = g["lr"], g["beta_1"], g["beta_2"], g["epsilon"]
        if "m" not in state:
            state["m"] = self._zeros(p)
            state["u"] = self._zeros(p)
        t = state["step"]
        _, eff, second = self._adaptive_effective_gradient(p, state, g)
        m, u = state["m"], state["u"]
        new_m = b1 * m + (1.0 - b1) * eff
        new_u = torch.maximum(b2 * u, second.abs())
        m.copy_(new_m)
        u.copy_(new_u)
        p.sub_(lr / (1.0 - b1 ** t) * new_m / (new_u + eps))

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        return self._run(closure, self._update)


__all__ = [
    "AdaptiveMemoryFSGD", "AdaptiveMemoryFRMSprop", "AdaptiveMemoryFAdam",
    "AdaptiveMemoryFAdadelta", "AdaptiveMemoryFAdagrad", "AdaptiveMemoryFAdamW",
    "AdaptiveMemoryFNadam", "AdaptiveMemoryFAdamax",
]


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(128, 10)
    y = torch.randint(0, 3, (128,))
    common = dict(vderiv=0.9, history_size=4, initial_mix=0.10, max_mix=0.30, adapt_step=0.01)
    configs = [
        ("AdaptiveMemoryFSGD", lambda ps: AdaptiveMemoryFSGD(ps, lr=1e-2, **common)),
        ("AdaptiveMemoryFRMSprop", lambda ps: AdaptiveMemoryFRMSprop(ps, lr=1e-3, **common)),
        ("AdaptiveMemoryFAdam", lambda ps: AdaptiveMemoryFAdam(ps, lr=1e-3, **common)),
        ("AdaptiveMemoryFAdadelta", lambda ps: AdaptiveMemoryFAdadelta(ps, lr=1.0, **common)),
        ("AdaptiveMemoryFAdagrad", lambda ps: AdaptiveMemoryFAdagrad(ps, lr=1e-2, **common)),
        ("AdaptiveMemoryFAdamW", lambda ps: AdaptiveMemoryFAdamW(ps, lr=1e-3, weight_decay=0.004, **common)),
        ("AdaptiveMemoryFNadam", lambda ps: AdaptiveMemoryFNadam(ps, lr=1e-3, **common)),
        ("AdaptiveMemoryFAdamax", lambda ps: AdaptiveMemoryFAdamax(ps, lr=1e-3, **common)),
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
        lam = float(torch.stack(opt.current_mix_values()).mean())
        print(f"  {name:24s} loss {losses[0]:.4f} -> {losses[-1]:.4f}  mean lambda {lam:.3f}")
        assert losses[-1] < losses[0], f"{name}: loss did not decrease"
    print("All smoke tests passed.")
