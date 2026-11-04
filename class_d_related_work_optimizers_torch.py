"""
class_d_related_work_optimizers_torch.py

PyTorch implementations (tested with PyTorch 2.15) of three published
fractional optimizer families: AdaGL, FCSGD_GL / FCAdam_GL and AOFGD.

This file is the PyTorch counterpart of `class_d_related_work_optimizers_keras3.py`:
same classes, same arguments, same update rules.

    ┌──────────────────────────────────────────────────────────────────────┐
    │ ORIGINAL WORK - these methods are not ours. If you use them, cite:    │
    │  AdaGL      S. Chen, C. Zhang, H. Mu, Neural Processing Letters 56,   │
    │             106 (2024), https://doi.org/10.1007/s11063-024-11571-7    │
    │  FCSGD_GL / FCAdam_GL                                                 │
    │             X. Zhou, C. Zhao, Y. Huang, Mathematics 11(2), 316 (2023),│
    │             https://doi.org/10.3390/math11020316                      │
    │  AOFGD      K. Xiang, C. Zhao, Q. Su, X. Zhou, J. Ye, Y. Huang,       │
    │             SSRN preprint 5717167 (2025),                             │
    │             https://doi.org/10.2139/ssrn.5717167                      │
    │ This file is an independent re-implementation from the papers, not   │
    │ the authors' code. Choices beyond the papers are noted below.         │
    └──────────────────────────────────────────────────────────────────────┘

Mathematical definitions (identical to the Keras version)
---------------------------------------------------------
AdaGL:
    w_0 = 1, w_j = (1 - (alpha+1)/j) w_{j-1};   g_frac_t = sum_{j=0}^{K-1} w_j g_{t-j}
    C_t = 1.1 - 0.5 / (1 + |g_t - g_{t-1}|)                       (element-wise)
    Adam moments from g_frac_t (both moments); theta -= lr * C_t * m_hat / (sqrt(v_hat) + eps)

FCSGD_GL / FCAdam_GL:
    g_frac_t = g_t + sum_{i=1}^{K-1} c[i-1] w_i g_{t-i},  c ~ Bernoulli(perturbation_prob)
    (K = history_size = 11 gives the 10 historical terms of Zhou et al., Eq. 10)
    FCSGD_GL: SGD (optional momentum) on g_frac_t;  FCAdam_GL: Adam with g_frac_t in both moments.

AOFGD:
    f_alpha(theta) = (|theta| + eps)^(1 - alpha) / Gamma(2 - alpha);  g_frac_t = g_t * f_{alpha_t}(theta_t)
    CEF_t = ||g_t|| / (||g_{t-1}|| + eps_cef);  alpha -= delta if CEF > 1 + thr,
    alpha += delta if CEF < 1 - thr; clip to [alpha_min, alpha_max]; EMA with ema_gamma.
    AOFGD_SGD: SGD (optional momentum) on g_frac_t.
    AOFGD_Adam: first moment from g_frac_t, second moment from the raw g_t
                (an implementation choice of this code base, as in the Keras version).

PyTorch notes
-------------
- `lr=` (alias `learning_rate=`), Keras argument names for everything else,
  hyperparameters per parameter group, state_dict() / load_state_dict().
- The Bernoulli masks use torch.rand on the parameter's device; pass
  `generator=torch.Generator(device).manual_seed(...)` for an independent,
  reproducible stream (the generator itself is not part of state_dict()).
- AOFGD keeps its smoothed order and previous gradient norm as scalars per
  parameter (the Keras version stores them as filled tensors only to stay
  bit-compatible with the legacy file; the values are the same).
"""

from __future__ import annotations

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


def _compute_gl_coefficients(alpha: float, n_terms: int) -> List[float]:
    """w_0 = 1, w_k = (1 - (alpha + 1)/k) w_{k-1}; [1, 0, ..., 0] at alpha = 1."""
    if abs(alpha - 1.0) < 1e-12:
        return [1.0] + [0.0] * (n_terms - 1)
    coeffs = [1.0]
    for k in range(1, n_terms):
        coeffs.append((1.0 - (alpha + 1.0) / k) * coeffs[-1])
    return coeffs


def _zeros(p):
    return torch.zeros_like(p, memory_format=torch.preserve_format)


class _ValidatingMixin:
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        super().add_param_group(param_group)
        self._validate_group(self.param_groups[-1])


def _check_history_group(g):
    _check(0.0 < float(g["alpha"]) < 2.0, "`alpha` must satisfy 0 < alpha < 2.")
    k = g["history_size"]
    _check(isinstance(k, int) and not isinstance(k, bool) and k >= 1,
           "`history_size` must be an integer >= 1.")
    _check(g["lr"] >= 0.0, "`lr` must be >= 0.")


def _shifted_history(state, raw, p, k):
    """History of shape p.shape + [K], index 0 = newest (same layout as Keras)."""
    if "grad_history" not in state:
        state["grad_history"] = torch.zeros(tuple(p.shape) + (k,), dtype=p.dtype, device=p.device)
    history = state["grad_history"]
    return torch.cat([raw.unsqueeze(-1), history[..., :-1]], dim=-1)


# =============================================================================
# Group 1: AdaGL  (Chen, Zhang & Mu, Neural Processing Letters 2024)
# =============================================================================


class AdaGL(_ValidatingMixin, torch.optim.Optimizer):
    """
    AdaGL: G-L fractional gradient (K terms) in both Adam moments, scaled by
    the element-wise step-size coefficient C_t = 1.1 - 0.5 / (1 + |g_t - g_{t-1}|).
    """

    def __init__(self, params: Iterable, lr: float = 0.001, alpha: float = 0.9,
                 history_size: int = 10, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-7, amsgrad: bool = False, coeff_epsilon: float = 1e-12,
                 learning_rate: Optional[float] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), alpha=float(alpha),
                        history_size=history_size, beta_1=float(beta_1), beta_2=float(beta_2),
                        epsilon=float(epsilon), amsgrad=bool(amsgrad),
                        coeff_epsilon=float(coeff_epsilon))
        super().__init__(params, defaults)

    def _validate_group(self, g):
        _check_history_group(g)
        _check(0.0 <= g["beta_1"] < 1.0, "`beta_1` must satisfy 0 <= beta_1 < 1.")
        _check(0.0 <= g["beta_2"] < 1.0, "`beta_2` must satisfy 0 <= beta_2 < 1.")
        _check(g["epsilon"] > 0.0, "`epsilon` must be > 0.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, b1, b2, eps, k = g["lr"], g["beta_1"], g["beta_2"], g["epsilon"], g["history_size"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["prev_grad"] = _zeros(p)
                    state["m"] = _zeros(p)
                    state["v"] = _zeros(p)
                    if g["amsgrad"]:
                        state["vhat"] = _zeros(p)
                state["step"] += 1
                t = state["step"]
                raw = _dense(p.grad).to(p.dtype)

                # step-size coefficient from the raw gradient difference
                c_t = 1.1 - 0.5 / (1.0 + (raw - state["prev_grad"]).abs())

                # GL fractional gradient
                new_history = _shifted_history(state, raw, p, k)
                if abs(g["alpha"] - 1.0) < 1e-12 or k == 1:
                    g_frac = raw
                else:
                    coeffs = torch.tensor(_compute_gl_coefficients(g["alpha"], k),
                                          dtype=p.dtype, device=p.device)
                    g_frac = (new_history * coeffs.view((1,) * p.dim() + (k,))).sum(dim=-1)

                m, v = state["m"], state["v"]
                new_m = b1 * m + (1.0 - b1) * g_frac
                new_v = b2 * v + (1.0 - b2) * g_frac * g_frac
                m.copy_(new_m)
                v.copy_(new_v)

                m_hat = new_m / (1.0 - b1 ** t)
                if g["amsgrad"]:
                    vhat = state["vhat"]
                    new_vhat = torch.maximum(vhat, new_v)
                    vhat.copy_(new_vhat)
                    v_hat = new_vhat / (1.0 - b2 ** t)
                else:
                    v_hat = new_v / (1.0 - b2 ** t)

                p.sub_(lr * c_t * m_hat / (v_hat.sqrt() + eps))
                state["grad_history"].copy_(new_history)
                state["prev_grad"].copy_(raw)
        return loss


# =============================================================================
# Group 2: FCSGD_GL and FCAdam_GL  (Zhou, Zhao & Huang, Mathematics 2023)
# =============================================================================


class _FCGLBase(_ValidatingMixin):
    """g_frac_t = g_t + sum_{i=1}^{K-1} c[i-1] w_i g_{t-i}, c ~ Bernoulli(perturbation_prob)."""

    def _validate_group(self, g):
        _check_history_group(g)
        _check(0.0 <= g["perturbation_prob"] <= 1.0, "`perturbation_prob` must be in [0, 1].")
        self._validate_base(g)

    def _fcgl_gradient(self, raw, p, state, g):
        k = g["history_size"]
        new_history = _shifted_history(state, raw, p, k)
        if abs(g["alpha"] - 1.0) < 1e-12 or k == 1:
            return new_history, raw
        n_past = k - 1
        past_history = new_history[..., 1:]
        past_coeffs = torch.tensor(_compute_gl_coefficients(g["alpha"], k)[1:],
                                   dtype=p.dtype, device=p.device)
        uniform = torch.rand(n_past, device=p.device, dtype=torch.float32,
                             generator=self._generator)
        mask = (uniform < g["perturbation_prob"]).to(p.dtype)
        effective = (mask * past_coeffs).view((1,) * p.dim() + (n_past,))
        g_frac = raw + (past_history * effective).sum(dim=-1)
        return new_history, g_frac


class FCSGD_GL(_FCGLBase, torch.optim.Optimizer):
    """FCSGD_GL: SGD (optional momentum / Nesterov) on the perturbed GL gradient."""

    def __init__(self, params: Iterable, lr: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False, alpha: float = 0.9, history_size: int = 11,
                 perturbation_prob: float = 0.5, coeff_epsilon: float = 1e-12,
                 learning_rate: Optional[float] = None,
                 generator: Optional[torch.Generator] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), momentum=float(momentum),
                        nesterov=bool(nesterov), alpha=float(alpha), history_size=history_size,
                        perturbation_prob=float(perturbation_prob),
                        coeff_epsilon=float(coeff_epsilon))
        self._generator = generator
        super().__init__(params, defaults)

    def _validate_base(self, g):
        _check(0.0 <= g["momentum"] < 1.0, "`momentum` must satisfy 0 <= momentum < 1.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, mu = g["lr"], g["momentum"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1
                raw = _dense(p.grad).to(p.dtype)
                new_history, g_frac = self._fcgl_gradient(raw, p, state, g)
                state["grad_history"].copy_(new_history)
                if mu > 0.0:
                    if "momentum" not in state:
                        state["momentum"] = _zeros(p)
                    m = state["momentum"]
                    new_m = mu * m - lr * g_frac
                    m.copy_(new_m)
                    if g["nesterov"]:
                        p.add_(mu * new_m - lr * g_frac)
                    else:
                        p.add_(new_m)
                else:
                    p.sub_(lr * g_frac)
        return loss


class FCAdam_GL(_FCGLBase, torch.optim.Optimizer):
    """FCAdam_GL: Adam with the perturbed GL gradient in BOTH moments."""

    def __init__(self, params: Iterable, lr: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, epsilon: float = 1e-7, amsgrad: bool = False,
                 alpha: float = 0.9, history_size: int = 11, perturbation_prob: float = 0.5,
                 coeff_epsilon: float = 1e-12, learning_rate: Optional[float] = None,
                 generator: Optional[torch.Generator] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), beta_1=float(beta_1),
                        beta_2=float(beta_2), epsilon=float(epsilon), amsgrad=bool(amsgrad),
                        alpha=float(alpha), history_size=history_size,
                        perturbation_prob=float(perturbation_prob),
                        coeff_epsilon=float(coeff_epsilon))
        self._generator = generator
        super().__init__(params, defaults)

    def _validate_base(self, g):
        _check(0.0 <= g["beta_1"] < 1.0, "`beta_1` must satisfy 0 <= beta_1 < 1.")
        _check(0.0 <= g["beta_2"] < 1.0, "`beta_2` must satisfy 0 <= beta_2 < 1.")
        _check(g["epsilon"] > 0.0, "`epsilon` must be > 0.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, b1, b2, eps = g["lr"], g["beta_1"], g["beta_2"], g["epsilon"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = _zeros(p)
                    state["v"] = _zeros(p)
                    if g["amsgrad"]:
                        state["vhat"] = _zeros(p)
                state["step"] += 1
                t = state["step"]
                raw = _dense(p.grad).to(p.dtype)
                new_history, g_frac = self._fcgl_gradient(raw, p, state, g)
                state["grad_history"].copy_(new_history)

                m, v = state["m"], state["v"]
                new_m = b1 * m + (1.0 - b1) * g_frac
                new_v = b2 * v + (1.0 - b2) * g_frac * g_frac
                m.copy_(new_m)
                v.copy_(new_v)
                m_hat = new_m / (1.0 - b1 ** t)
                if g["amsgrad"]:
                    vhat = state["vhat"]
                    new_vhat = torch.maximum(vhat, new_v)
                    vhat.copy_(new_vhat)
                    v_hat = new_vhat / (1.0 - b2 ** t)
                else:
                    v_hat = new_v / (1.0 - b2 ** t)
                p.sub_(lr * m_hat / (v_hat.sqrt() + eps))
        return loss


# =============================================================================
# Group 3: AOFGD family  (Xiang, Zhao, Su, Zhou, Ye & Huang, SSRN preprint 2025)
# =============================================================================


class _AOFGDBase(_ValidatingMixin):
    """Adaptive-order Caputo-type gradient scaling with a CEF-driven order."""

    @staticmethod
    def _aofgd_defaults(alpha_init, alpha_min, alpha_max, delta_alpha, cef_threshold,
                        ema_gamma, frac_epsilon, eps_cef) -> Dict[str, Any]:
        return dict(alpha_init=float(alpha_init), alpha_min=float(alpha_min),
                    alpha_max=float(alpha_max), delta_alpha=float(delta_alpha),
                    cef_threshold=float(cef_threshold), ema_gamma=float(ema_gamma),
                    frac_epsilon=float(frac_epsilon), eps_cef=float(eps_cef))

    def _validate_group(self, g):
        _check(0.0 < g["alpha_init"] < 2.0, "`alpha_init` must satisfy 0 < alpha_init < 2.")
        _check(0.0 < g["alpha_min"] <= g["alpha_init"], "`alpha_min` must satisfy 0 < alpha_min <= alpha_init.")
        _check(g["alpha_init"] <= g["alpha_max"] < 2.0, "`alpha_max` must satisfy alpha_init <= alpha_max < 2.")
        _check(g["delta_alpha"] >= 0.0, "`delta_alpha` must be >= 0.")
        _check(0.0 <= g["ema_gamma"] < 1.0, "`ema_gamma` must satisfy 0 <= ema_gamma < 1.")
        _check(g["frac_epsilon"] > 0.0, "`frac_epsilon` must be > 0.")
        _check(g["eps_cef"] > 0.0, "`eps_cef` must be > 0.")
        _check(g["lr"] >= 0.0, "`lr` must be >= 0.")
        self._validate_base(g)

    @staticmethod
    def _init_aofgd_state(p, state, g):
        state["step"] = 0
        state["alpha_smooth"] = torch.full((), g["alpha_init"], dtype=p.dtype, device=p.device)
        # small positive start value keeps CEF finite in the first step (as in Keras)
        state["prev_grad_norm"] = torch.full((), 1e-4, dtype=p.dtype, device=p.device)

    @staticmethod
    def _update_alpha(raw, state, g):
        alpha = state["alpha_smooth"]
        prev_norm = state["prev_grad_norm"]
        curr_norm = ((raw * raw).sum() + 1e-30).sqrt()
        cef = curr_norm / (prev_norm + g["eps_cef"])
        thr, d = g["cef_threshold"], g["delta_alpha"]
        new_raw = torch.where(cef > 1.0 + thr, alpha - d,
                              torch.where(cef < 1.0 - thr, alpha + d, alpha))
        new_raw = torch.clamp(new_raw, g["alpha_min"], g["alpha_max"])
        new_smooth = g["ema_gamma"] * alpha + (1.0 - g["ema_gamma"]) * new_raw
        alpha.copy_(new_smooth)
        prev_norm.copy_(curr_norm)
        return new_smooth

    @staticmethod
    def _caputo_gradient(raw, theta, alpha, g):
        """g * (|theta| + eps)^(1 - alpha) / Gamma(2 - alpha), Gamma via lgamma."""
        gamma_val = torch.exp(torch.lgamma(2.0 - alpha))
        return raw * (torch.pow(theta.abs() + g["frac_epsilon"], 1.0 - alpha) / gamma_val)

    def current_alpha_values(self) -> List[torch.Tensor]:
        """Current smoothed fractional order, one scalar per parameter."""
        out = []
        for g in self.param_groups:
            for p in g["params"]:
                st = self.state.get(p, {})
                if "alpha_smooth" in st:
                    out.append(st["alpha_smooth"].detach().clone())
        return out


class AOFGD_SGD(_AOFGDBase, torch.optim.Optimizer):
    """AOFGD_SGD: SGD (optional momentum / Nesterov) on the adaptive-order Caputo gradient."""

    def __init__(self, params: Iterable, lr: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False, alpha_init: float = 0.9, alpha_min: float = 0.5,
                 alpha_max: float = 1.9, delta_alpha: float = 0.02, cef_threshold: float = 0.05,
                 ema_gamma: float = 0.95, frac_epsilon: float = 1e-6, eps_cef: float = 1e-8,
                 learning_rate: Optional[float] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), momentum=float(momentum),
                        nesterov=bool(nesterov),
                        **self._aofgd_defaults(alpha_init, alpha_min, alpha_max, delta_alpha,
                                               cef_threshold, ema_gamma, frac_epsilon, eps_cef))
        super().__init__(params, defaults)

    def _validate_base(self, g):
        _check(0.0 <= g["momentum"] < 1.0, "`momentum` must satisfy 0 <= momentum < 1.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, mu = g["lr"], g["momentum"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    self._init_aofgd_state(p, state, g)
                state["step"] += 1
                raw = _dense(p.grad).to(p.dtype)
                theta = p.detach().clone()
                alpha = self._update_alpha(raw, state, g)
                g_frac = self._caputo_gradient(raw, theta, alpha, g)
                if mu > 0.0:
                    if "momentum" not in state:
                        state["momentum"] = _zeros(p)
                    m = state["momentum"]
                    new_m = mu * m - lr * g_frac
                    m.copy_(new_m)
                    if g["nesterov"]:
                        p.add_(mu * new_m - lr * g_frac)
                    else:
                        p.add_(new_m)
                else:
                    p.sub_(lr * g_frac)
        return loss


class AOFGD_Adam(_AOFGDBase, torch.optim.Optimizer):
    """AOFGD_Adam: first moment from the adaptive-order gradient, second from the raw gradient."""

    def __init__(self, params: Iterable, lr: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, epsilon: float = 1e-7, amsgrad: bool = False,
                 alpha_init: float = 0.9, alpha_min: float = 0.5, alpha_max: float = 1.9,
                 delta_alpha: float = 0.02, cef_threshold: float = 0.05, ema_gamma: float = 0.95,
                 frac_epsilon: float = 1e-6, eps_cef: float = 1e-8,
                 learning_rate: Optional[float] = None):
        defaults = dict(lr=_resolve_lr(lr, learning_rate), beta_1=float(beta_1),
                        beta_2=float(beta_2), epsilon=float(epsilon), amsgrad=bool(amsgrad),
                        **self._aofgd_defaults(alpha_init, alpha_min, alpha_max, delta_alpha,
                                               cef_threshold, ema_gamma, frac_epsilon, eps_cef))
        super().__init__(params, defaults)

    def _validate_base(self, g):
        _check(0.0 <= g["beta_1"] < 1.0, "`beta_1` must satisfy 0 <= beta_1 < 1.")
        _check(0.0 <= g["beta_2"] < 1.0, "`beta_2` must satisfy 0 <= beta_2 < 1.")
        _check(g["epsilon"] > 0.0, "`epsilon` must be > 0.")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, b1, b2, eps = g["lr"], g["beta_1"], g["beta_2"], g["epsilon"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    self._init_aofgd_state(p, state, g)
                    state["m"] = _zeros(p)
                    state["v"] = _zeros(p)
                    if g["amsgrad"]:
                        state["vhat"] = _zeros(p)
                state["step"] += 1
                t = state["step"]
                raw = _dense(p.grad).to(p.dtype)
                theta = p.detach().clone()
                alpha = self._update_alpha(raw, state, g)
                g_frac = self._caputo_gradient(raw, theta, alpha, g)

                m, v = state["m"], state["v"]
                new_m = b1 * m + (1.0 - b1) * g_frac
                new_v = b2 * v + (1.0 - b2) * raw * raw
                m.copy_(new_m)
                v.copy_(new_v)
                m_hat = new_m / (1.0 - b1 ** t)
                if g["amsgrad"]:
                    vhat = state["vhat"]
                    new_vhat = torch.maximum(vhat, new_v)
                    vhat.copy_(new_vhat)
                    v_hat = new_vhat / (1.0 - b2 ** t)
                else:
                    v_hat = new_v / (1.0 - b2 ** t)
                p.sub_(lr * m_hat / (v_hat.sqrt() + eps))
        return loss


__all__ = ["AdaGL", "FCSGD_GL", "FCAdam_GL", "AOFGD_SGD", "AOFGD_Adam"]


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(128, 10)
    y = torch.randint(0, 3, (128,))
    configs = [
        ("AdaGL", lambda ps: AdaGL(ps, lr=1e-3, alpha=0.9, history_size=10)),
        ("FCSGD_GL", lambda ps: FCSGD_GL(ps, lr=1e-2, alpha=0.9, history_size=11, perturbation_prob=0.5)),
        ("FCAdam_GL", lambda ps: FCAdam_GL(ps, lr=1e-3, alpha=0.9, history_size=11, perturbation_prob=0.5)),
        ("AOFGD_SGD", lambda ps: AOFGD_SGD(ps, lr=1e-2, alpha_init=0.9, alpha_min=0.5, alpha_max=1.9)),
        ("AOFGD_Adam", lambda ps: AOFGD_Adam(ps, lr=1e-3, alpha_init=0.9, alpha_min=0.5, alpha_max=1.9)),
    ]
    for name, make in configs:
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
        opt = make(model.parameters())
        for epoch in range(2):
            perm = torch.randperm(128)
            for i in range(0, 128, 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                torch.nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
                opt.step()
        with torch.no_grad():
            final = float(torch.nn.functional.cross_entropy(model(x), y))
        print(f"  {name:12s} loss={final:.4f}  OK")
    print("All smoke tests passed.")
