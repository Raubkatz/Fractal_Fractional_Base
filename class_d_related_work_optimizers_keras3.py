"""
class_d_related_work_optimizers_keras3.py

Keras 3 / TensorFlow >= 2.16 implementations of the three published
fractional optimizer families most closely related to the Adaptive
Memory-Fractional framework.

This file is the Keras 3 version of `class_d_related_work_optimizers.py`.
The update rules are the same, operation for operation (including the
storage layout of the gradient history and the per-variable state of AOFGD,
so that summation order and rounding are unchanged). Only the optimizer
plumbing changes, from `optimizer_v2` (`_set_hyper`, `_create_slots`,
`_resource_apply_dense`) to the Keras 3 API (`build`, `update_step`).

Backend: TensorFlow (the Keras 3 default when Keras is installed together
with TensorFlow).

Implemented optimizers
----------------------
Group 1: AdaGL (Chen, Zhang & Mu, 2024)
    - AdaGL

Group 2: FCSGD_GL / FCAdam_GL (Zhou, Zhao & Huang, 2023)
    - FCSGD_GL
    - FCAdam_GL

Group 3: AOFGD family (Xiang, Zhao, Su, Zhou, Ye & Huang, 2025)
    - AOFGD_SGD
    - AOFGD_Adam

Mathematical definitions
------------------------

--- AdaGL ---

GL fractional gradient (K=10 terms, order alpha), recursive coefficients:
    w_0 = 1
    w_j = (1 - (alpha+1)/j) * w_{j-1},  j >= 1
    g_frac_t = sum_{j=0}^{K-1} w_j * g_{t-j}

Step-size control coefficient (element-wise):
    C_t = 1.1 - 0.5 / (1 + |g_t - g_{t-1}|),   C_t in [0.6, 1.1)

Adam moments using the fractional gradient:
    m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_frac_t
    v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_frac_t^2
    theta_t = theta_{t-1} - eta * C_t * m_hat_t / (sqrt(v_hat_t) + eps)

--- FCSGD_GL / FCAdam_GL ---

GL fractional gradient with stochastic perturbation (Zhou et al., 2023):
    g_frac_t = g_t + sum_{i=1}^{K-1} c[i-1] * w_i * g_{t-i}
    c[i-1] ~ Bernoulli(perturbation_prob), resampled every step
The default history_size=11 gives the 10 historical terms of Zhou et al.
(2023), Eq. 10. perturbation_prob=1.0 disables the perturbation.

FCSGD_GL:   theta_t = theta_{t-1} - eta * g_frac_t   (optional momentum)
FCAdam_GL:  Adam with g_frac_t in BOTH moments.

--- AOFGD family ---

Adaptive-order Caputo-type gradient (Herrera-style scaling, time-varying order):
    f_alpha(theta) = (|theta| + eps)^(1 - alpha) / Gamma(2 - alpha)
    g_frac_t = g_t * f_{alpha_t}(theta_t)

Convergence Evaluation Factor:
    CEF_t = ||g_t||_2 / (||g_{t-1}||_2 + eps_cef)
Order rule:
    CEF_t > 1 + thr  ->  alpha_t = alpha_{t-1} - delta_alpha
    CEF_t < 1 - thr  ->  alpha_t = alpha_{t-1} + delta_alpha
    otherwise        ->  alpha_t = alpha_{t-1}
    alpha_t = clip(alpha_t, alpha_min, alpha_max), followed by an EMA with
    ema_gamma; the smoothed order is used in f_alpha.

AOFGD_SGD:  theta_t = theta_{t-1} - eta * g_frac_t   (optional momentum)
AOFGD_Adam: first moment from g_frac_t, second moment from the raw g_t.

Compatibility
-------------
- `learning_rate=` is the Keras 3 argument; `lr=` is accepted as an alias.
- Keras 3 base-optimizer arguments pass through **kwargs.
- All classes are registered for `.keras` model saving/loading.
- Legacy optimizer checkpoints cannot be restored into these classes.
- The Bernoulli masks of FCSGD_GL / FCAdam_GL use tf.random.uniform, as in
  the legacy file; set keras.utils.set_random_seed(...) for repeatable runs.

References
----------
1. Chen, S.; Zhang, C.; Mu, H. An Adaptive Learning Rate Deep Learning
   Optimizer Using Long and Short-Term Gradients Based on G-L Fractional-
   Order Derivative. Neural Processing Letters 2024, 56, 106.
   https://doi.org/10.1007/s11063-024-11571-7

2. Zhou, X.; Zhao, C.; Huang, Y. A Deep Learning Optimizer Based on
   Grunwald-Letnikov Fractional Order Definition. Mathematics 2023, 11(2),
   316. https://doi.org/10.3390/math11020316

3. Xiang, K.; Zhao, C.; Su, Q.; Zhou, X.; Ye, J.; Huang, Y. AOFGD:
   Adaptive Order Fractional Gradient Descent Method. Preprint, 2025.
   https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5717167

Usage example
-------------
    from class_d_related_work_optimizers_keras3 import AdaGL, FCAdam_GL, AOFGD_Adam

    optimizer = AdaGL(learning_rate=1e-3, alpha=0.9, history_size=10)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
    model.fit(x, y, epochs=10)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import keras
import tensorflow as tf

if keras.backend.backend() != "tensorflow":  # pragma: no cover
    raise ImportError(
        "class_d_related_work_optimizers_keras3 requires the TensorFlow "
        "backend of Keras 3 (KERAS_BACKEND=tensorflow). Current backend: "
        f"{keras.backend.backend()!r}."
    )


# =============================================================================
# Small helpers
# =============================================================================


def _resolve_learning_rate(learning_rate, lr):
    """Accept the legacy `lr=` alias; Keras 3 requires a Python float."""
    value = learning_rate if lr is None else lr
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
    return value


def _dtype(variable):
    """Base dtype of a model variable (tf.Variable or keras.Variable)."""
    return tf.as_dtype(variable.dtype).base_dtype


def _read(variable):
    """Read the current value of an optimizer/model variable as a tensor."""
    return tf.convert_to_tensor(variable)


def _variable_name(variable, suffix):
    base = getattr(variable, "path", None) or str(variable.name)
    return base.replace("/", "_").replace(":", "_") + "_" + suffix


def _compute_gl_coefficients(alpha: float, n_terms: int) -> List[float]:
    """
    Compute the short-memory Grunwald-Letnikov coefficient sequence.

    Recursion:
        w_0 = 1
        w_k = (1 - (alpha + 1) / k) * w_{k-1},  k >= 1

    For alpha numerically equal to 1.0, returns [1, 0, ..., 0].
    """
    if abs(alpha - 1.0) < 1e-12:
        return [1.0] + [0.0] * (n_terms - 1)
    coeffs = [1.0]
    for k in range(1, n_terms):
        coeffs.append((1.0 - (alpha + 1.0) / k) * coeffs[-1])
    return coeffs


class _HistoryMixin:
    """
    Gradient history stored as ONE variable of shape var.shape + [K]
    (index 0 = newest). This is the same layout as the legacy
    `tf.stack(K slots, axis=-1)`, so the GL sums are computed in the same
    order. The whole history is shifted by one every step (as in the legacy
    file).
    """

    def _build_history(self, var_list):
        self._histories = []
        for var in var_list:
            colocate = var.value if isinstance(var, keras.Variable) else var
            with self._distribution_strategy.extended.colocate_vars_with(colocate):
                self._histories.append(
                    self.add_variable(
                        shape=tuple(var.shape) + (self.history_size,),
                        initializer="zeros",
                        dtype=var.dtype,
                        name=_variable_name(var, "grad_history"),
                    )
                )

    def _history_var(self, var):
        return self._histories[self._get_variable_index(var)]

    def _to_dense(self, grad, var):
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                tf.expand_dims(grad.indices, 1),
                grad.values,
                tf.shape(var),
            )
        return grad

    def _all_reduce_sum_gradients(self, grads_and_vars):
        """
        Single-replica shortcut. Without a multi-replica tf.distribute
        strategy the all-reduce is the identity, so the gradients are passed
        through unchanged. This avoids a TensorFlow memory leak: the default
        all-reduce registers a tf.custom_gradient function in a global
        registry on every trace, which keeps each traced training graph alive
        (noticeable when many models are trained in one process). With
        several replicas, the normal Keras all-reduce is used.
        """
        replica_context = tf.distribute.get_replica_context()
        if replica_context is None or replica_context.num_replicas_in_sync == 1:
            return list(grads_and_vars)
        return super()._all_reduce_sum_gradients(grads_and_vars)


# =============================================================================
# Group 1: AdaGL
# Chen, Zhang & Mu, Neural Processing Letters 2024
# =============================================================================


@keras.saving.register_keras_serializable(package="RelatedWorkFractional")
class AdaGL(_HistoryMixin, keras.optimizers.Optimizer):
    """
    AdaGL: Adaptive Learning Rate Deep Learning Optimizer Using Long and
    Short-Term Gradients Based on G-L Fractional-Order Derivative.

    Reference: Chen, Zhang & Mu (2024), Neural Processing Letters, 56, 106.

    Algorithm:
        1. Compute G-L fractional gradient g_frac_t using history of length K.
        2. Compute step-size control coefficient C_t from short-term gradient
           change (element-wise).
        3. Update Adam moments using g_frac_t for BOTH first and second moments.
        4. Apply bias-corrected Adam update scaled by C_t.

    The step-size control coefficient is:
        C_t = 1.1 - 0.5 / (1 + |g_t - g_{t-1}|),   C_t in [0.6, 1.1)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        alpha: float = 0.9,
        history_size: int = 10,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        coeff_epsilon: float = 1e-12,
        name: str = "AdaGL",
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=_resolve_learning_rate(learning_rate, lr),
            name=name,
            **kwargs,
        )

        if not (0.0 < float(alpha) < 2.0):
            raise ValueError("`alpha` must satisfy 0 < alpha < 2.")
        if not isinstance(history_size, int) or history_size < 1:
            raise ValueError("`history_size` must be an integer >= 1.")
        if not (0.0 <= beta_1 < 1.0):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError("`beta_2` must satisfy 0 <= beta_2 < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.alpha = float(alpha)
        self.history_size = int(history_size)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = bool(amsgrad)
        self.coeff_epsilon = float(coeff_epsilon)

        self._gl_coeffs = _compute_gl_coefficients(self.alpha, self.history_size)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_history(var_list)
        self._prev_grads, self._m, self._v, self._vhat = [], [], [], []
        for var in var_list:
            self._prev_grads.append(self.add_variable_from_reference(var, name="prev_grad"))
            self._m.append(self.add_variable_from_reference(var, name="m"))
            self._v.append(self.add_variable_from_reference(var, name="v"))
            if self.amsgrad:
                self._vhat.append(self.add_variable_from_reference(var, name="vhat"))

    def _gl_fractional_gradient(self, raw_grad, var):
        """
        Compute the GL fractional gradient and the shifted history.

        Returns (new_history, g_frac).
        """
        dtype = _dtype(var)
        history = _read(self._history_var(var))

        # Shift history: prepend current gradient, drop oldest
        new_history = tf.concat(
            [tf.expand_dims(raw_grad, axis=-1), history[..., :-1]],
            axis=-1,
        )

        if abs(self.alpha - 1.0) < 1e-12 or self.history_size == 1:
            return new_history, raw_grad

        coeffs = tf.constant(self._gl_coeffs, dtype=dtype)
        reshape = [1] * len(var.shape) + [self.history_size]
        coeffs = tf.reshape(coeffs, reshape)
        g_frac = tf.reduce_sum(new_history * coeffs, axis=-1)
        return new_history, g_frac

    def _step_size_coefficient(self, raw_grad, prev_grad, dtype):
        """
        Element-wise step-size control coefficient
        C_t = 1.1 - 0.5 / (1 + |g_t - g_{t-1}|).
        """
        diff = tf.abs(tf.cast(raw_grad, dtype) - tf.cast(prev_grad, dtype))
        c = tf.cast(1.1, dtype) - tf.cast(0.5, dtype) / (
            tf.cast(1.0, dtype) + diff
        )
        return c

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        raw_grad = tf.cast(self._to_dense(gradient, variable), dtype)
        prev_grad_var = self._prev_grads[idx]
        m = self._m[idx]
        v = self._v[idx]

        # Step-size coefficient uses the raw gradient difference
        c_t = self._step_size_coefficient(raw_grad, _read(prev_grad_var), dtype)

        # GL fractional gradient
        new_history, g_frac = self._gl_fractional_gradient(raw_grad, variable)

        # Update moments with fractional gradient (both m and v)
        new_m = b1 * _read(m) + (1.0 - b1) * g_frac
        new_v = b2 * _read(v) + (1.0 - b2) * tf.square(g_frac)

        self.assign(m, new_m)
        self.assign(v, new_v)

        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)
        m_hat = new_m / (1.0 - b1_pow)

        if self.amsgrad:
            vhat = self._vhat[idx]
            new_vhat = tf.maximum(_read(vhat), new_v)
            self.assign(vhat, new_vhat)
            v_hat = new_vhat / (1.0 - b2_pow)
        else:
            v_hat = new_v / (1.0 - b2_pow)

        # C_t scales the learning rate element-wise
        self.assign_sub(variable, lr_t * c_t * m_hat / (tf.sqrt(v_hat) + eps))

        self.assign(self._history_var(variable), new_history)
        self.assign(prev_grad_var, raw_grad)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "alpha": self.alpha,
                "history_size": self.history_size,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "coeff_epsilon": self.coeff_epsilon,
            }
        )
        return cfg


# =============================================================================
# Group 2: FCSGD_GL and FCAdam_GL
# Zhou, Zhao & Huang, Mathematics 2023
# =============================================================================


class _FCGLBase(_HistoryMixin):
    """
    Shared GL gradient construction for FCSGD_GL and FCAdam_GL.

        g_frac_t = g_t + sum_{i=1}^{K-1} c[i-1] * w_i * g_{t-i},
        c[i-1] ~ Bernoulli(perturbation_prob).

    With the default history_size=11 the sum covers the 10 historical terms
    of Zhou et al. (2023), Eq. 10. The GL coefficients use the standard
    recursion w_0 = 1, w_k = (1 - (alpha+1)/k) w_{k-1}.
    """

    def _init_fcgl_args(
        self,
        alpha: float = 0.9,
        history_size: int = 10,
        perturbation_prob: float = 0.5,
        coeff_epsilon: float = 1e-12,
    ) -> None:
        if not (0.0 < float(alpha) < 2.0):
            raise ValueError("`alpha` must satisfy 0 < alpha < 2.")
        if not isinstance(history_size, int) or history_size < 1:
            raise ValueError("`history_size` must be an integer >= 1.")
        if not (0.0 <= perturbation_prob <= 1.0):
            raise ValueError("`perturbation_prob` must be in [0, 1].")

        self.alpha = float(alpha)
        self.history_size = int(history_size)
        self.perturbation_prob = float(perturbation_prob)
        self.coeff_epsilon = float(coeff_epsilon)

        # Full coefficient vector w_0..w_{K-1}
        self._gl_coeffs = _compute_gl_coefficients(self.alpha, self.history_size)

    def _fcgl_gradient(self, raw_grad, var):
        """
        Compute the stochastically perturbed GL fractional gradient.

        Returns (new_history, g_frac). The Bernoulli mask is sampled fresh
        on every call.
        """
        dtype = _dtype(var)
        history = _read(self._history_var(var))

        # Shift history
        new_history = tf.concat(
            [tf.expand_dims(raw_grad, axis=-1), history[..., :-1]],
            axis=-1,
        )

        if abs(self.alpha - 1.0) < 1e-12 or self.history_size == 1:
            return new_history, raw_grad

        # w_0 = 1 corresponds to the current gradient (already included as g_t).
        past_history = new_history[..., 1:]  # shape: var_shape + [K-1]
        past_coeffs = self._gl_coeffs[1:]    # w_1 .. w_{K-1}

        # Bernoulli perturbation mask: shape [K-1], broadcast over var dims
        n_past = self.history_size - 1
        mask = tf.cast(
            tf.less(
                tf.random.uniform([n_past], dtype=tf.float32),
                tf.constant(self.perturbation_prob, dtype=tf.float32),
            ),
            dtype=dtype,
        )  # each entry is 1 with prob p, 0 otherwise

        coeffs_tensor = tf.constant(past_coeffs, dtype=dtype)  # [K-1]
        effective_coeffs = mask * coeffs_tensor                 # [K-1]

        reshape = [1] * len(var.shape) + [n_past]
        effective_coeffs = tf.reshape(effective_coeffs, reshape)

        past_contribution = tf.reduce_sum(past_history * effective_coeffs, axis=-1)
        g_frac = raw_grad + past_contribution

        return new_history, g_frac

    def _fcgl_config(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "history_size": self.history_size,
            "perturbation_prob": self.perturbation_prob,
            "coeff_epsilon": self.coeff_epsilon,
        }


@keras.saving.register_keras_serializable(package="RelatedWorkFractional")
class FCSGD_GL(_FCGLBase, keras.optimizers.Optimizer):
    """
    FCSGD_GL: Fractional Calculus SGD based on the G-L definition.

    Reference: Zhou, Zhao & Huang, Mathematics 2023, 11(2), 316.

    Update:
        g_frac_t = g_t + sum_{i=1}^{10} c[i-1] * w_i * g_{t-i}
        theta_{t+1} = theta_t - eta * g_frac_t        (optional momentum)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lr: float = None,
        momentum: float = 0.0,
        nesterov: bool = False,
        alpha: float = 0.9,
        history_size: int = 11,
        perturbation_prob: float = 0.5,
        coeff_epsilon: float = 1e-12,
        name: str = "FCSGD_GL",
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=_resolve_learning_rate(learning_rate, lr),
            name=name,
            **kwargs,
        )

        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")

        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self._init_fcgl_args(
            alpha=alpha,
            history_size=history_size,
            perturbation_prob=perturbation_prob,
            coeff_epsilon=coeff_epsilon,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_history(var_list)
        self._momentums = []
        if self.momentum > 0.0:
            for var in var_list:
                self._momentums.append(self.add_variable_from_reference(var, name="momentum"))

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        lr_t = tf.cast(learning_rate, dtype)
        raw_grad = tf.cast(self._to_dense(gradient, variable), dtype)

        new_history, g_frac = self._fcgl_gradient(raw_grad, variable)
        self.assign(self._history_var(variable), new_history)

        if self.momentum > 0.0:
            mu = tf.cast(self.momentum, dtype)
            m = self._momentums[self._get_variable_index(variable)]
            new_m = mu * _read(m) - lr_t * g_frac
            self.assign(m, new_m)
            if self.nesterov:
                self.assign_add(variable, mu * new_m - lr_t * g_frac)
            else:
                self.assign_add(variable, new_m)
            return

        self.assign_sub(variable, lr_t * g_frac)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **self._fcgl_config(),
            }
        )
        return cfg


@keras.saving.register_keras_serializable(package="RelatedWorkFractional")
class FCAdam_GL(_FCGLBase, keras.optimizers.Optimizer):
    """
    FCAdam_GL: Fractional Calculus Adam based on the G-L definition.

    Reference: Zhou, Zhao & Huang, Mathematics 2023, 11(2), 316.

    The GL fractional gradient (with Bernoulli perturbation) enters BOTH
    the first and second Adam moment estimates:
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_frac_t
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_frac_t^2
        theta_{t+1} = theta_t - eta * m_hat_t / (sqrt(v_hat_t) + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        alpha: float = 0.9,
        history_size: int = 11,
        perturbation_prob: float = 0.5,
        coeff_epsilon: float = 1e-12,
        name: str = "FCAdam_GL",
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=_resolve_learning_rate(learning_rate, lr),
            name=name,
            **kwargs,
        )

        if not (0.0 <= beta_1 < 1.0):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError("`beta_2` must satisfy 0 <= beta_2 < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = bool(amsgrad)
        self._init_fcgl_args(
            alpha=alpha,
            history_size=history_size,
            perturbation_prob=perturbation_prob,
            coeff_epsilon=coeff_epsilon,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_history(var_list)
        self._m, self._v, self._vhat = [], [], []
        for var in var_list:
            self._m.append(self.add_variable_from_reference(var, name="m"))
            self._v.append(self.add_variable_from_reference(var, name="v"))
            if self.amsgrad:
                self._vhat.append(self.add_variable_from_reference(var, name="vhat"))

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        raw_grad = tf.cast(self._to_dense(gradient, variable), dtype)
        new_history, g_frac = self._fcgl_gradient(raw_grad, variable)
        self.assign(self._history_var(variable), new_history)

        m = self._m[idx]
        v = self._v[idx]

        # Both moments use the fractional gradient (original paper design)
        new_m = b1 * _read(m) + (1.0 - b1) * g_frac
        new_v = b2 * _read(v) + (1.0 - b2) * tf.square(g_frac)

        self.assign(m, new_m)
        self.assign(v, new_v)

        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)
        m_hat = new_m / (1.0 - b1_pow)

        if self.amsgrad:
            vhat = self._vhat[idx]
            new_vhat = tf.maximum(_read(vhat), new_v)
            self.assign(vhat, new_vhat)
            v_hat = new_vhat / (1.0 - b2_pow)
        else:
            v_hat = new_v / (1.0 - b2_pow)

        self.assign_sub(variable, lr_t * m_hat / (tf.sqrt(v_hat) + eps))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._fcgl_config(),
            }
        )
        return cfg


# =============================================================================
# Group 3: AOFGD family
# Xiang, Zhao, Su, Zhou, Ye & Huang, SSRN Preprint 2025
# =============================================================================


class _AOFGDBase:
    """
    Shared adaptive-order Caputo-type gradient construction for AOFGD.

        f_alpha(theta) = (|theta| + eps_frac)^(1 - alpha) / Gamma(2 - alpha)
        g_frac_t = g_t * f_{alpha_t}(theta_t)

        CEF_t = ||g_t||_2 / (||g_{t-1}||_2 + eps_cef)
        ratchet on alpha with delta_alpha, clip to [alpha_min, alpha_max],
        EMA smoothing with ema_gamma.

    The smoothed order and the previous gradient norm are stored per model
    variable as variable-shaped tensors filled with the scalar value, as in
    the legacy file (this keeps the reduce_mean read-back, and therefore the
    rounding, identical).
    """

    def _init_aofgd_args(
        self,
        alpha_init: float = 0.9,
        alpha_min: float = 0.5,
        alpha_max: float = 1.9,
        delta_alpha: float = 0.02,
        cef_threshold: float = 0.05,
        ema_gamma: float = 0.95,
        frac_epsilon: float = 1e-6,
        eps_cef: float = 1e-8,
    ) -> None:
        if not (0.0 < float(alpha_init) < 2.0):
            raise ValueError("`alpha_init` must satisfy 0 < alpha_init < 2.")
        if not (0.0 < float(alpha_min) <= float(alpha_init)):
            raise ValueError("`alpha_min` must satisfy 0 < alpha_min <= alpha_init.")
        if not (float(alpha_init) <= float(alpha_max) < 2.0):
            raise ValueError("`alpha_max` must satisfy alpha_init <= alpha_max < 2.")
        if delta_alpha < 0.0:
            raise ValueError("`delta_alpha` must be >= 0.")
        if not (0.0 <= ema_gamma < 1.0):
            raise ValueError("`ema_gamma` must satisfy 0 <= ema_gamma < 1.")
        if frac_epsilon <= 0.0:
            raise ValueError("`frac_epsilon` must be > 0.")
        if eps_cef <= 0.0:
            raise ValueError("`eps_cef` must be > 0.")

        self.alpha_init = float(alpha_init)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.delta_alpha = float(delta_alpha)
        self.cef_threshold = float(cef_threshold)
        self.ema_gamma = float(ema_gamma)
        self.frac_epsilon = float(frac_epsilon)
        self.eps_cef = float(eps_cef)

        self._gamma_cache = {}

    def _gamma_factor(self, alpha_val: float) -> float:
        """Return Gamma(2 - alpha) with a small cache (Python-side helper)."""
        if alpha_val not in self._gamma_cache:
            self._gamma_cache[alpha_val] = float(math.gamma(2.0 - alpha_val))
        return self._gamma_cache[alpha_val]

    def _build_aofgd_variables(self, var_list):
        """Create the smoothed-order and previous-gradient-norm state."""
        self._alpha_smooth = []
        self._prev_grad_norms = []
        for var in var_list:
            self._alpha_smooth.append(
                self.add_variable_from_reference(
                    var,
                    name="alpha_smooth",
                    initializer=keras.initializers.Constant(self.alpha_init),
                )
            )
            # Initialized to a small positive value so that CEF = curr/prev is
            # finite on the first step (as in the legacy file).
            self._prev_grad_norms.append(
                self.add_variable_from_reference(
                    var,
                    name="prev_grad_norm",
                    initializer=keras.initializers.Constant(1e-4),
                )
            )

    def _to_dense(self, grad, var):
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                tf.expand_dims(grad.indices, 1),
                grad.values,
                tf.shape(var),
            )
        return grad

    def _scalar_from_slot(self, slot, dtype):
        return tf.cast(tf.reduce_mean(_read(slot)), dtype)

    def _fill_like(self, scalar, var):
        dtype = _dtype(var)
        return tf.ones_like(var, dtype=dtype) * tf.cast(scalar, dtype)

    def _compute_caputo_gradient(self, raw_grad, var, alpha_val_scalar):
        """
        f_alpha(theta) = (|theta| + eps)^(1 - alpha) / Gamma(2 - alpha)
        g_frac = g * f_alpha(theta), with Gamma evaluated in-graph via lgamma.
        """
        dtype = _dtype(var)
        eps = tf.cast(self.frac_epsilon, dtype)
        exponent = 1.0 - alpha_val_scalar
        log_gamma = tf.math.lgamma(
            tf.cast(2.0, dtype) - tf.cast(alpha_val_scalar, dtype)
        )
        gamma_val = tf.exp(log_gamma)

        factor = tf.pow(tf.abs(var) + eps, tf.cast(exponent, dtype)) / gamma_val
        return raw_grad * factor

    def _update_alpha(self, raw_grad, var, dtype):
        """
        Update the fractional order alpha based on the CEF and write the new
        state. Returns the new smoothed alpha (scalar tensor).
        """
        idx = self._get_variable_index(var)
        prev_norm_slot = self._prev_grad_norms[idx]
        alpha_slot = self._alpha_smooth[idx]

        alpha_curr = self._scalar_from_slot(alpha_slot, dtype)
        prev_norm = self._scalar_from_slot(prev_norm_slot, dtype)

        # Convergence evaluation factor
        curr_norm = tf.cast(
            tf.sqrt(tf.reduce_sum(tf.square(raw_grad)) + 1e-30), dtype
        )
        cef = curr_norm / (prev_norm + tf.cast(self.eps_cef, dtype))

        thr = tf.cast(self.cef_threshold, dtype)
        d_alpha = tf.cast(self.delta_alpha, dtype)

        # Ratchet rule
        new_alpha_raw = tf.cond(
            cef > tf.cast(1.0, dtype) + thr,
            true_fn=lambda: alpha_curr - d_alpha,
            false_fn=lambda: tf.cond(
                cef < tf.cast(1.0, dtype) - thr,
                true_fn=lambda: alpha_curr + d_alpha,
                false_fn=lambda: alpha_curr,
            ),
        )
        new_alpha_raw = tf.clip_by_value(
            new_alpha_raw,
            tf.cast(self.alpha_min, dtype),
            tf.cast(self.alpha_max, dtype),
        )

        # EMA smoothing
        gamma = tf.cast(self.ema_gamma, dtype)
        new_alpha_smooth = gamma * alpha_curr + (1.0 - gamma) * new_alpha_raw

        self.assign(alpha_slot, self._fill_like(new_alpha_smooth, var))
        self.assign(prev_norm_slot, self._fill_like(curr_norm, var))
        return new_alpha_smooth

    def current_alpha_values(self) -> List[tf.Tensor]:
        """Return the current smoothed fractional order for each variable."""
        return [tf.reduce_mean(_read(w)) for w in getattr(self, "_alpha_smooth", [])]

    def _aofgd_config(self) -> Dict[str, Any]:
        return {
            "alpha_init": self.alpha_init,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "delta_alpha": self.delta_alpha,
            "cef_threshold": self.cef_threshold,
            "ema_gamma": self.ema_gamma,
            "frac_epsilon": self.frac_epsilon,
            "eps_cef": self.eps_cef,
        }

    def _all_reduce_sum_gradients(self, grads_and_vars):
        """
        Single-replica shortcut. Without a multi-replica tf.distribute
        strategy the all-reduce is the identity, so the gradients are passed
        through unchanged. This avoids a TensorFlow memory leak: the default
        all-reduce registers a tf.custom_gradient function in a global
        registry on every trace, which keeps each traced training graph alive
        (noticeable when many models are trained in one process). With
        several replicas, the normal Keras all-reduce is used.
        """
        replica_context = tf.distribute.get_replica_context()
        if replica_context is None or replica_context.num_replicas_in_sync == 1:
            return list(grads_and_vars)
        return super()._all_reduce_sum_gradients(grads_and_vars)


@keras.saving.register_keras_serializable(package="RelatedWorkFractional")
class AOFGD_SGD(_AOFGDBase, keras.optimizers.Optimizer):
    """
    AOFGD_SGD: Adaptive Order Fractional Gradient Descent with SGD base.

    Reference: Xiang et al., SSRN Preprint 2025.

    Update:
        CEF_t = ||g_t|| / ||g_{t-1}||
        alpha_t adapted via ratchet + EMA smoothing
        g_frac_t = g_t * (|theta_t| + eps)^(1 - alpha_t) / Gamma(2 - alpha_t)
        theta_{t+1} = theta_t - eta * g_frac_t
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lr: float = None,
        momentum: float = 0.0,
        nesterov: bool = False,
        alpha_init: float = 0.9,
        alpha_min: float = 0.5,
        alpha_max: float = 1.9,
        delta_alpha: float = 0.02,
        cef_threshold: float = 0.05,
        ema_gamma: float = 0.95,
        frac_epsilon: float = 1e-6,
        eps_cef: float = 1e-8,
        name: str = "AOFGD_SGD",
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=_resolve_learning_rate(learning_rate, lr),
            name=name,
            **kwargs,
        )

        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")

        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self._init_aofgd_args(
            alpha_init=alpha_init,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            delta_alpha=delta_alpha,
            cef_threshold=cef_threshold,
            ema_gamma=ema_gamma,
            frac_epsilon=frac_epsilon,
            eps_cef=eps_cef,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_aofgd_variables(var_list)
        self._momentums = []
        if self.momentum > 0.0:
            for var in var_list:
                self._momentums.append(self.add_variable_from_reference(var, name="momentum"))

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        lr_t = tf.cast(learning_rate, dtype)
        raw_grad = tf.cast(self._to_dense(gradient, variable), dtype)

        # Fractional scaling uses the weights before this step's update
        theta = _read(variable)

        # Adapt the fractional order
        alpha_smooth = self._update_alpha(raw_grad, variable, dtype)

        # Apply Caputo-type fractional scaling
        g_frac = self._compute_caputo_gradient(raw_grad, theta, alpha_smooth)

        if self.momentum > 0.0:
            mu = tf.cast(self.momentum, dtype)
            m = self._momentums[self._get_variable_index(variable)]
            new_m = mu * _read(m) - lr_t * g_frac
            self.assign(m, new_m)
            if self.nesterov:
                self.assign_add(variable, mu * new_m - lr_t * g_frac)
            else:
                self.assign_add(variable, new_m)
            return

        self.assign_sub(variable, lr_t * g_frac)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **self._aofgd_config(),
            }
        )
        return cfg


@keras.saving.register_keras_serializable(package="RelatedWorkFractional")
class AOFGD_Adam(_AOFGDBase, keras.optimizers.Optimizer):
    """
    AOFGD_Adam: Adaptive Order Fractional Gradient Descent with Adam base.

    Reference: Xiang et al., SSRN Preprint 2025.

    Update:
        CEF_t = ||g_t|| / ||g_{t-1}||
        alpha_t adapted via ratchet + EMA smoothing
        g_frac_t = g_t * (|theta_t| + eps)^(1 - alpha_t) / Gamma(2 - alpha_t)
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_frac_t
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2       [raw gradient]
        theta_{t+1} = theta_t - eta * m_hat_t / (sqrt(v_hat_t) + eps_adam)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        alpha_init: float = 0.9,
        alpha_min: float = 0.5,
        alpha_max: float = 1.9,
        delta_alpha: float = 0.02,
        cef_threshold: float = 0.05,
        ema_gamma: float = 0.95,
        frac_epsilon: float = 1e-6,
        eps_cef: float = 1e-8,
        name: str = "AOFGD_Adam",
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=_resolve_learning_rate(learning_rate, lr),
            name=name,
            **kwargs,
        )

        if not (0.0 <= beta_1 < 1.0):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError("`beta_2` must satisfy 0 <= beta_2 < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = bool(amsgrad)
        self._init_aofgd_args(
            alpha_init=alpha_init,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            delta_alpha=delta_alpha,
            cef_threshold=cef_threshold,
            ema_gamma=ema_gamma,
            frac_epsilon=frac_epsilon,
            eps_cef=eps_cef,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_aofgd_variables(var_list)
        self._m, self._v, self._vhat = [], [], []
        for var in var_list:
            self._m.append(self.add_variable_from_reference(var, name="m"))
            self._v.append(self.add_variable_from_reference(var, name="v"))
            if self.amsgrad:
                self._vhat.append(self.add_variable_from_reference(var, name="vhat"))

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        raw_grad = tf.cast(self._to_dense(gradient, variable), dtype)
        theta = _read(variable)

        # Adapt the fractional order
        alpha_smooth = self._update_alpha(raw_grad, variable, dtype)

        # Fractional gradient for first moment
        g_frac = self._compute_caputo_gradient(raw_grad, theta, alpha_smooth)

        m = self._m[idx]
        v = self._v[idx]

        # First moment uses fractional gradient
        new_m = b1 * _read(m) + (1.0 - b1) * g_frac
        # Second moment uses raw gradient for stability
        new_v = b2 * _read(v) + (1.0 - b2) * tf.square(raw_grad)

        self.assign(m, new_m)
        self.assign(v, new_v)

        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)
        m_hat = new_m / (1.0 - b1_pow)

        if self.amsgrad:
            vhat = self._vhat[idx]
            new_vhat = tf.maximum(_read(vhat), new_v)
            self.assign(vhat, new_vhat)
            v_hat = new_vhat / (1.0 - b2_pow)
        else:
            v_hat = new_v / (1.0 - b2_pow)

        self.assign_sub(variable, lr_t * m_hat / (tf.sqrt(v_hat) + eps))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._aofgd_config(),
            }
        )
        return cfg


# =============================================================================
# Export list
# =============================================================================

__all__ = [
    # Group 1: AdaGL
    "AdaGL",
    # Group 2: FCSGD_GL / FCAdam_GL
    "FCSGD_GL",
    "FCAdam_GL",
    # Group 3: AOFGD
    "AOFGD_SGD",
    "AOFGD_Adam",
]


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("Running smoke tests for class_d_related_work_optimizers_keras3.py")
    print("=" * 60)

    keras.utils.set_random_seed(42)

    x = tf.random.normal((128, 10))
    y = tf.random.uniform((128,), minval=0, maxval=3, dtype=tf.int32)

    configs = [
        ("AdaGL", AdaGL(learning_rate=1e-3, alpha=0.9, history_size=10)),
        ("FCSGD_GL", FCSGD_GL(learning_rate=1e-2, alpha=0.9, history_size=11, perturbation_prob=0.5)),
        ("FCAdam_GL", FCAdam_GL(learning_rate=1e-3, alpha=0.9, history_size=11, perturbation_prob=0.5)),
        ("AOFGD_SGD", AOFGD_SGD(learning_rate=1e-2, alpha_init=0.9, alpha_min=0.5, alpha_max=1.9)),
        ("AOFGD_Adam", AOFGD_Adam(learning_rate=1e-3, alpha_init=0.9, alpha_min=0.5, alpha_max=1.9)),
    ]

    for name, opt in configs:
        model = keras.Sequential(
            [
                keras.Input(shape=(10,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        hist = model.fit(x, y, epochs=2, batch_size=32, verbose=0)
        final_loss = hist.history["loss"][-1]
        final_acc = hist.history["accuracy"][-1]
        print(f"  {name:20s}  loss={final_loss:.4f}  acc={final_acc:.4f}  OK")

    print("=" * 60)
    print("All smoke tests passed.")
