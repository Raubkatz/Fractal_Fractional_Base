"""
class_c_tf_fractional_optimizers_SR_keras3.py

Explicit memory-based fractional (Grunwald-Letnikov) optimizers for
TensorFlow >= 2.16 / Keras 3.

This file is the Keras 3 version of `class_c_tf_fractional_optimizers_SR.py`
(the corrected and extended version with circular history buffer, norm
matching, descent safeguard and raw-gradient second moments). The update
rules are the same, operation for operation. Only the optimizer plumbing
changes, from `optimizer_v2` (`_set_hyper`, `_create_slots`,
`_resource_apply_dense`) to the Keras 3 API (`build`, `update_step`).

Backend: TensorFlow (the Keras 3 default when Keras is installed together
with TensorFlow). The update steps use TensorFlow ops directly so that the
arithmetic matches the legacy implementation exactly.

Definitions (canonical notation of the article)
-----------------------------------------------
Raw gradient            g_t = dL/dtheta_t
GL coefficients         c_0^(nu)=1,  c_k^(nu) = (1-(nu+1)/k) c_{k-1}^(nu)
Fractional gradient     g_t^(nu) = sum_{k=0}^{min(t,K-1)} c_k^(nu) g_{t-k}
Descent safeguard       g_t^(nu) <- g_t            if <g_t^(nu), g_t> <= 0
Norm matching           g_hat_t^(nu) = g_t^(nu) ||g_t|| / (||g_t^(nu)||+eps)
Update                  base-optimizer update with g_hat_t^(nu) as the
                        direction input; squared-gradient accumulators from
                        g_t (default) or from g_hat_t^(nu).

Design choices (all switchable, defaults as in the legacy corrected file)
-------------------------------------------------------------------------
norm_match=True
    The truncated GL kernel has a small DC gain (sum of coefficients), e.g.
    +0.018 for nu=0.9, K=8. Rescaling the fractional gradient to the norm of
    the raw gradient keeps the step size on the scale the base optimizer was
    tuned for; the memory term contributes the direction.
descent_check=True
    For nu > 1 the truncated coefficient sum is negative (e.g. -0.016 for
    nu=1.5, K=8), so under near-constant gradients the memory direction can
    point uphill. If <g^(nu), g_t> <= 0 for a variable, the raw gradient is
    used for that variable in that step.
use_raw_gradient_for_second_moment=True
    All squared-gradient accumulators are driven by the raw gradient g_t; the
    fractional gradient enters only the update direction / first moment.
    Set to False to feed the fractional gradient into both moments.
Circular history buffer
    One variable of shape [K] + var.shape per parameter; one row is written
    per step (position t mod K) and the GL coefficients are rotated with
    tf.gather instead of moving the data.
Classical fallback
    For |nu - 1| < 1e-12 or K == 1 the history machinery is skipped and each
    optimizer is identical to its base optimizer.

Implemented optimizers
----------------------
    MemoryFSGD          (SGD, optional momentum / Nesterov)
    MemoryFRMSprop      (RMSprop, optional momentum / centered)
    MemoryFAdam         (Adam, optional AMSGrad)
    MemoryFAdadelta     (Adadelta)
    MemoryFAdagrad      (Adagrad)
    MemoryFAdamW        (AdamW, decoupled weight decay after the Adam step)
    MemoryFNadam        (Nadam, Nesterov-Adam)
    MemoryFAdamax       (Adamax, infinity-norm second moment)

Compatibility
-------------
- `learning_rate=` is the Keras 3 argument; `lr=` is accepted as an alias.
- Keras 3 base-optimizer arguments (clipnorm, clipvalue, global_clipnorm,
  use_ema, loss_scale_factor, gradient_accumulation_steps, weight_decay, ...)
  pass through **kwargs.
- All classes are registered for `.keras` model saving/loading.
- Legacy optimizer checkpoints cannot be restored into these classes.

Usage
-----
    from class_c_tf_fractional_optimizers_SR_keras3 import MemoryFAdam

    optimizer = MemoryFAdam(learning_rate=1e-3, vderiv=0.9, history_size=6)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
    model.fit(x, y, epochs=10)

References
----------
1. Zhou, X.; Zhao, C.; Huang, Y. A Deep Learning Optimizer Based on
   Grunwald-Letnikov Fractional Order Definition. Mathematics 2023, 11, 316.
2. Podlubny, I. Fractional Differential Equations. Academic Press, 1999.
"""

from __future__ import annotations

from typing import Any, Dict, List

import keras
import tensorflow as tf

if keras.backend.backend() != "tensorflow":  # pragma: no cover
    raise ImportError(
        "class_c_tf_fractional_optimizers_SR_keras3 requires the TensorFlow "
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


# =============================================================================
# Shared helper base
# =============================================================================


class _MemoryFractionalBase:
    """
    Shared helper functionality for explicit memory-based fractional
    optimizers with a circular gradient-history buffer.
    """

    _CLASSICAL_TOL = 1e-12

    # ------------------------------------------------------------------
    # Init and coefficients
    # ------------------------------------------------------------------

    def _init_memory_fractional_args(
        self,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        norm_match: bool = True,
        use_raw_gradient_for_second_moment: bool = True,
        descent_check: bool = True,
    ) -> None:
        if not isinstance(vderiv, (float, int)):
            raise TypeError("`vderiv` must be a float.")
        if not (0.0 < float(vderiv) < 2.0):
            raise ValueError("`vderiv` must satisfy 0 < vderiv < 2.")
        if not isinstance(history_size, int) or history_size < 1:
            raise ValueError("`history_size` must be an integer >= 1.")
        if coeff_epsilon <= 0.0:
            raise ValueError("`coeff_epsilon` must be > 0.")

        self.vderiv = float(vderiv)
        self.history_size = int(history_size)
        self.normalize_coefficients = bool(normalize_coefficients)
        self.coeff_epsilon = float(coeff_epsilon)
        self.norm_match = bool(norm_match)
        self.use_raw_gradient_for_second_moment = bool(
            use_raw_gradient_for_second_moment
        )
        self.descent_check = bool(descent_check)

        self._fractional_coefficients = self._compute_fractional_coefficients(
            self.vderiv,
            self.history_size,
            self.normalize_coefficients,
        )

    def _compute_fractional_coefficients(
        self,
        nu: float,
        n_terms: int,
        normalize: bool = False,
    ) -> List[float]:
        """
        Short-memory Grunwald-Letnikov coefficients.

        c_0 = 1
        c_k = (1 - (nu + 1)/k) * c_{k-1},  k >= 1

        Classical-limit convention: if nu is numerically 1, [1, 0, ..., 0].
        """
        if abs(nu - 1.0) < self._CLASSICAL_TOL:
            return [1.0] + [0.0] * (n_terms - 1)

        coeffs = [1.0]
        for k in range(1, n_terms):
            coeffs.append((1.0 - (nu + 1.0) / k) * coeffs[-1])

        if normalize:
            denom = sum(abs(c) for c in coeffs)
            if denom < self.coeff_epsilon:
                denom = 1.0
            coeffs = [c / denom for c in coeffs]

        return coeffs

    def gl_coefficient_sum(self) -> float:
        """
        Diagnostic: sum of the (possibly normalized) GL coefficients.
        This is the steady-state ("DC") gain of the memory filter before
        norm matching. Values near zero or negative indicate strong
        cancellation of the raw filter; norm matching compensates the
        magnitude, the descent safeguard the sign.
        """
        return float(sum(self._fractional_coefficients))

    @property
    def gl_coefficients(self) -> List[float]:
        """The GL coefficients c_0 ... c_{K-1} used by this optimizer."""
        return list(self._fractional_coefficients)

    @property
    def _is_classical(self) -> bool:
        return (
            abs(self.vderiv - 1.0) < self._CLASSICAL_TOL
            or self.history_size == 1
        )

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------

    def _to_dense_gradient(self, grad, var):
        """Convert sparse gradients to dense tensors."""
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                indices=tf.expand_dims(grad.indices, axis=1),
                updates=grad.values,
                shape=tf.shape(var),
            )
        return grad

    def _global_l2_norm(self, x, dtype):
        x = tf.cast(x, dtype)
        return tf.sqrt(tf.reduce_sum(tf.square(x)))

    # ------------------------------------------------------------------
    # Circular history buffer (one variable per parameter)
    # ------------------------------------------------------------------

    def _add_shaped_variable(self, reference, shape, suffix, initializer="zeros"):
        """Optimizer variable with a custom shape, colocated with `reference`."""
        colocate = reference.value if isinstance(reference, keras.Variable) else reference
        with self._distribution_strategy.extended.colocate_vars_with(colocate):
            return self.add_variable(
                shape=tuple(shape),
                initializer=initializer,
                dtype=reference.dtype,
                name=_variable_name(reference, suffix),
            )

    def _build_history_variables(self, var_list) -> None:
        """One variable of shape [K] + var.shape per parameter (circular buffer)."""
        self._grad_histories = []
        for var in var_list:
            if self._is_classical:
                self._grad_histories.append(None)
            else:
                self._grad_histories.append(
                    self._add_shaped_variable(
                        var,
                        (self.history_size,) + tuple(var.shape),
                        "grad_history",
                    )
                )

    def _fractional_gradient(self, raw_grad, var):
        """
        Write the current gradient into the circular buffer and return the
        corrected fractional gradient.

        Steps:
          1. row  <- t mod K,   buffer[row] <- g_t          (single write)
          2. g^(nu) = sum_j c_{lag(j)} buffer[j],
             lag(j) = (row - j) mod K                       (gather, no copy)
          3. descent safeguard: fall back to g_t if <g^(nu), g_t> <= 0
          4. norm matching:     rescale to ||g_t||
        """
        dtype = _dtype(var)

        if self._is_classical:
            return raw_grad

        k = self.history_size
        hist_var = self._grad_histories[self._get_variable_index(var)]

        # (1) single-row write at the circular position of this step.
        # self.iterations is the 0-based global step during update_step.
        pos = tf.cast(
            tf.math.floormod(tf.cast(self.iterations, tf.int64), k), tf.int32
        )
        self.assign(
            hist_var,
            tf.IndexedSlices(
                values=tf.expand_dims(raw_grad, axis=0),
                indices=tf.reshape(pos, [1]),
            ),
        )
        # Read after the write (TensorFlow orders the two resource ops).
        hist = _read(hist_var)

        # (2) rotate coefficients instead of shifting data.
        coeffs = tf.constant(self._fractional_coefficients, dtype=dtype)
        lags = tf.math.floormod(pos - tf.range(k, dtype=tf.int32), k)
        c_perm = tf.gather(coeffs, lags)                     # [K]
        c_shape = [k] + [1] * len(var.shape)
        frac_grad = tf.reduce_sum(
            hist * tf.reshape(c_perm, c_shape), axis=0
        )

        # (3) descent safeguard (per variable).
        if self.descent_check:
            dot = tf.reduce_sum(frac_grad * raw_grad)
            keep = tf.cast(dot > 0.0, dtype)
            frac_grad = keep * frac_grad + (1.0 - keep) * raw_grad

        # (4) norm matching.
        if self.norm_match:
            raw_norm = self._global_l2_norm(raw_grad, dtype)
            frac_norm = self._global_l2_norm(frac_grad, dtype)
            scale = raw_norm / (frac_norm + tf.cast(self.coeff_epsilon, dtype))
            frac_grad = frac_grad * scale

        return frac_grad

    def _direction_and_second_moment_grads(self, grad, var):
        """
        Returns (raw_grad, direction_grad, second_moment_grad).
        """
        raw_grad = tf.cast(self._to_dense_gradient(grad, var), _dtype(var))
        frac_grad = self._fractional_gradient(raw_grad, var)
        second_grad = (
            raw_grad if self.use_raw_gradient_for_second_moment else frac_grad
        )
        return raw_grad, frac_grad, second_grad

    def _memory_fractional_config(self) -> Dict[str, Any]:
        return {
            "vderiv": self.vderiv,
            "history_size": self.history_size,
            "normalize_coefficients": self.normalize_coefficients,
            "coeff_epsilon": self.coeff_epsilon,
            "norm_match": self.norm_match,
            "use_raw_gradient_for_second_moment":
                self.use_raw_gradient_for_second_moment,
            "descent_check": self.descent_check,
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


# =============================================================================
# MemoryFSGD
# =============================================================================


@keras.saving.register_keras_serializable(package="MemoryFractional")
class MemoryFSGD(_MemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Explicit memory-based fractional SGD.

    Update (no momentum):
        theta_{t+1} = theta_t - lr * g_hat_t^(nu)
    Update (momentum):
        v_t = momentum * v_{t-1} - lr * g_hat_t^(nu)
        theta_{t+1} = theta_t + v_t          (Nesterov variant as in Keras)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lr: float = None,
        momentum: float = 0.0,
        nesterov: bool = False,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        norm_match: bool = True,
        use_raw_gradient_for_second_moment: bool = True,
        descent_check: bool = True,
        name: str = "MemoryFSGD",
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

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
            norm_match=norm_match,
            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
            descent_check=descent_check,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_history_variables(var_list)
        self._momentums = []
        if self.momentum > 0.0:
            for var in var_list:
                self._momentums.append(
                    self.add_variable_from_reference(var, name="momentum")
                )

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        lr_t = tf.cast(learning_rate, dtype)

        _, frac_grad, _ = self._direction_and_second_moment_grads(gradient, variable)

        if self.momentum > 0.0:
            mu = tf.cast(self.momentum, dtype)
            m = self._momentums[self._get_variable_index(variable)]
            new_m = mu * _read(m) - lr_t * frac_grad
            self.assign(m, new_m)
            if self.nesterov:
                self.assign_add(variable, mu * new_m - lr_t * frac_grad)
            else:
                self.assign_add(variable, new_m)
            return

        self.assign_sub(variable, lr_t * frac_grad)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# MemoryFRMSprop
# =============================================================================


@keras.saving.register_keras_serializable(package="MemoryFractional")
class MemoryFRMSprop(_MemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Explicit memory-based fractional RMSprop.

    The fractional gradient drives the update direction; the squared-
    gradient accumulator uses the raw gradient by default.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        centered: bool = False,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        norm_match: bool = True,
        use_raw_gradient_for_second_moment: bool = True,
        descent_check: bool = True,
        name: str = "MemoryFRMSprop",
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=_resolve_learning_rate(learning_rate, lr),
            name=name,
            **kwargs,
        )

        if not (0.0 <= rho < 1.0):
            raise ValueError("`rho` must satisfy 0 <= rho < 1.")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.rho = float(rho)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.centered = bool(centered)

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
            norm_match=norm_match,
            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
            descent_check=descent_check,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_history_variables(var_list)
        self._velocities, self._momentums, self._average_grads = [], [], []
        for var in var_list:
            self._velocities.append(self.add_variable_from_reference(var, name="velocity"))
            if self.momentum > 0.0:
                self._momentums.append(self.add_variable_from_reference(var, name="momentum"))
            if self.centered:
                self._average_grads.append(self.add_variable_from_reference(var, name="avg_grad"))

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, dtype)
        rho_t = tf.cast(self.rho, dtype)
        eps_t = tf.cast(self.epsilon, dtype)

        _, frac_grad, second_grad = self._direction_and_second_moment_grads(
            gradient, variable
        )

        velocity = self._velocities[idx]
        new_velocity = rho_t * _read(velocity) + (1.0 - rho_t) * tf.square(second_grad)
        self.assign(velocity, new_velocity)

        if self.centered:
            avg_grad = self._average_grads[idx]
            new_avg = rho_t * _read(avg_grad) + (1.0 - rho_t) * second_grad
            self.assign(avg_grad, new_avg)
            denom = tf.sqrt(new_velocity - tf.square(new_avg) + eps_t)
        else:
            denom = tf.sqrt(new_velocity + eps_t)

        if self.momentum > 0.0:
            mu = tf.cast(self.momentum, dtype)
            mom = self._momentums[idx]
            new_mom = mu * _read(mom) + lr_t * frac_grad / denom
            self.assign(mom, new_mom)
            self.assign_sub(variable, new_mom)
            return

        self.assign_sub(variable, lr_t * frac_grad / denom)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# MemoryFAdam
# =============================================================================


@keras.saving.register_keras_serializable(package="MemoryFractional")
class MemoryFAdam(_MemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Explicit memory-based fractional Adam.

    First moment from the fractional gradient; second moment from the raw
    gradient by default (set use_raw_gradient_for_second_moment=False to
    feed the fractional gradient into both moments).

    Keras-exact Adam formulation: the bias correction is folded into the
    learning rate and epsilon sits outside the uncorrected sqrt, so the
    nu = 1 fallback matches the standard Keras Adam update.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        norm_match: bool = True,
        use_raw_gradient_for_second_moment: bool = True,
        descent_check: bool = True,
        name: str = "MemoryFAdam",
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

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
            norm_match=norm_match,
            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
            descent_check=descent_check,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_history_variables(var_list)
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
        eps_t = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        _, frac_grad, second_grad = self._direction_and_second_moment_grads(
            gradient, variable
        )

        m = self._m[idx]
        v = self._v[idx]

        new_m = b1 * _read(m) + (1.0 - b1) * frac_grad
        new_v = b2 * _read(v) + (1.0 - b2) * tf.square(second_grad)

        self.assign(m, new_m)
        self.assign(v, new_v)

        # Keras-exact formulation: bias correction folded into the learning
        # rate, epsilon outside the *uncorrected* sqrt.
        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)
        lr_corr = lr_t * tf.sqrt(1.0 - b2_pow) / (1.0 - b1_pow)

        if self.amsgrad:
            vhat = self._vhat[idx]
            new_vhat = tf.maximum(_read(vhat), new_v)
            self.assign(vhat, new_vhat)
            denom_v = new_vhat
        else:
            denom_v = new_v

        self.assign_sub(variable, lr_corr * new_m / (tf.sqrt(denom_v) + eps_t))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# MemoryFAdadelta
# =============================================================================


@keras.saving.register_keras_serializable(package="MemoryFractional")
class MemoryFAdadelta(_MemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Explicit memory-based fractional Adadelta.

    Direction from the fractional gradient; both accumulators from the raw
    gradient / raw-driven update by default.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        lr: float = None,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        norm_match: bool = True,
        use_raw_gradient_for_second_moment: bool = True,
        descent_check: bool = True,
        name: str = "MemoryFAdadelta",
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=_resolve_learning_rate(learning_rate, lr),
            name=name,
            **kwargs,
        )

        if not (0.0 <= rho < 1.0):
            raise ValueError("`rho` must satisfy 0 <= rho < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.rho = float(rho)
        self.epsilon = float(epsilon)

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
            norm_match=norm_match,
            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
            descent_check=descent_check,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_history_variables(var_list)
        self._accumulated_grads, self._accumulated_deltas = [], []
        for var in var_list:
            self._accumulated_grads.append(
                self.add_variable_from_reference(var, name="accumulated_grad")
            )
            self._accumulated_deltas.append(
                self.add_variable_from_reference(var, name="accumulated_delta")
            )

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, dtype)
        rho_t = tf.cast(self.rho, dtype)
        eps_t = tf.cast(self.epsilon, dtype)

        _, frac_grad, second_grad = self._direction_and_second_moment_grads(
            gradient, variable
        )

        acc_g = self._accumulated_grads[idx]
        acc_d = self._accumulated_deltas[idx]
        acc_d_value = _read(acc_d)

        new_acc_g = rho_t * _read(acc_g) + (1.0 - rho_t) * tf.square(second_grad)
        self.assign(acc_g, new_acc_g)

        rms_delta = tf.sqrt(acc_d_value + eps_t)
        rms_grad = tf.sqrt(new_acc_g + eps_t)
        delta = -(rms_delta / rms_grad) * frac_grad

        new_acc_d = rho_t * acc_d_value + (1.0 - rho_t) * tf.square(delta)
        self.assign(acc_d, new_acc_d)

        self.assign_add(variable, lr_t * delta)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rho": self.rho,
                "epsilon": self.epsilon,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# MemoryFAdagrad
# =============================================================================


@keras.saving.register_keras_serializable(package="MemoryFractional")
class MemoryFAdagrad(_MemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Explicit memory-based fractional Adagrad.

    Update:
        acc_t = acc_{t-1} + s_t^2          (s_t = raw gradient by default)
        theta_{t+1} = theta_t - lr * g_hat_t^(nu) / (sqrt(acc_t) + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        norm_match: bool = True,
        use_raw_gradient_for_second_moment: bool = True,
        descent_check: bool = True,
        name: str = "MemoryFAdagrad",
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=_resolve_learning_rate(learning_rate, lr),
            name=name,
            **kwargs,
        )

        if initial_accumulator_value < 0.0:
            raise ValueError("`initial_accumulator_value` must be >= 0.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.initial_accumulator_value = float(initial_accumulator_value)
        self.epsilon = float(epsilon)

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
            norm_match=norm_match,
            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
            descent_check=descent_check,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_history_variables(var_list)
        init = keras.initializers.Constant(self.initial_accumulator_value)
        self._accumulators = [
            self.add_variable_from_reference(var, name="accumulator", initializer=init)
            for var in var_list
        ]

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        lr_t = tf.cast(learning_rate, dtype)
        eps_t = tf.cast(self.epsilon, dtype)

        _, frac_grad, second_grad = self._direction_and_second_moment_grads(
            gradient, variable
        )

        acc = self._accumulators[self._get_variable_index(variable)]
        new_acc = _read(acc) + tf.square(second_grad)
        self.assign(acc, new_acc)

        self.assign_sub(variable, lr_t * frac_grad / (tf.sqrt(new_acc) + eps_t))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_accumulator_value": self.initial_accumulator_value,
                "epsilon": self.epsilon,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# MemoryFAdamW
# =============================================================================


@keras.saving.register_keras_serializable(package="MemoryFractional")
class MemoryFAdamW(MemoryFAdam):
    """
    Explicit memory-based fractional AdamW: MemoryFAdam plus decoupled
    weight decay applied after the Adam step, on the updated weights,
        theta' = theta_t - lr * (Adam step)
        theta_{t+1} = theta' - lr * weight_decay * theta'.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        weight_decay: float = 0.004,
        name: str = "MemoryFAdamW",
        **kwargs: Any,
    ):
        if weight_decay is None or weight_decay < 0.0:
            raise ValueError("`weight_decay` must be >= 0.")
        super().__init__(
            learning_rate=learning_rate, lr=lr, name=name, **kwargs
        )
        self.weight_decay = float(weight_decay)

    def _apply_weight_decay(self, variables):
        # Disabled on purpose: the decay is applied after the Adam step inside
        # update_step(), as in the legacy implementation.
        return

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        lr_t = tf.cast(learning_rate, dtype)
        super().update_step(gradient, variable, learning_rate)
        if self.weight_decay > 0.0 and self._use_weight_decay(variable):
            wd_t = tf.cast(self.weight_decay, dtype)
            self.assign_sub(variable, lr_t * wd_t * _read(variable))

    def get_config(self):
        config = super().get_config()
        config.update({"weight_decay": self.weight_decay})
        return config


# =============================================================================
# MemoryFNadam
# =============================================================================


@keras.saving.register_keras_serializable(package="MemoryFractional")
class MemoryFNadam(_MemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Explicit memory-based fractional Nadam (Nesterov-accelerated Adam).

    Update (Dozat's formulation without the momentum-product schedule):
        m_t = beta_1 m_{t-1} + (1 - beta_1) g_hat_t^(nu)
        v_t = beta_2 v_{t-1} + (1 - beta_2) s_t^2
        m_bar = beta_1 m_t / (1 - beta_1^{t+1})
                + (1 - beta_1) g_hat_t^(nu) / (1 - beta_1^t)
        theta_{t+1} = theta_t - lr * m_bar / (sqrt(v_t/(1-beta_2^t)) + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        norm_match: bool = True,
        use_raw_gradient_for_second_moment: bool = True,
        descent_check: bool = True,
        name: str = "MemoryFNadam",
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

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
            norm_match=norm_match,
            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
            descent_check=descent_check,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_history_variables(var_list)
        self._m, self._v = [], []
        for var in var_list:
            self._m.append(self.add_variable_from_reference(var, name="m"))
            self._v.append(self.add_variable_from_reference(var, name="v"))

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps_t = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        _, frac_grad, second_grad = self._direction_and_second_moment_grads(
            gradient, variable
        )

        m = self._m[idx]
        v = self._v[idx]

        new_m = b1 * _read(m) + (1.0 - b1) * frac_grad
        new_v = b2 * _read(v) + (1.0 - b2) * tf.square(second_grad)

        self.assign(m, new_m)
        self.assign(v, new_v)

        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)

        m_bar = (
            b1 * new_m / (1.0 - b1_pow * b1)
            + (1.0 - b1) * frac_grad / (1.0 - b1_pow)
        )
        v_hat = new_v / (1.0 - b2_pow)

        self.assign_sub(variable, lr_t * m_bar / (tf.sqrt(v_hat) + eps_t))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# MemoryFAdamax
# =============================================================================


@keras.saving.register_keras_serializable(package="MemoryFractional")
class MemoryFAdamax(_MemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Explicit memory-based fractional Adamax (infinity-norm Adam).

    Update:
        m_t = beta_1 m_{t-1} + (1 - beta_1) g_hat_t^(nu)
        u_t = max(beta_2 u_{t-1}, |s_t|)
        theta_{t+1} = theta_t - lr / (1 - beta_1^t) * m_t / (u_t + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        norm_match: bool = True,
        use_raw_gradient_for_second_moment: bool = True,
        descent_check: bool = True,
        name: str = "MemoryFAdamax",
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

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
            norm_match=norm_match,
            use_raw_gradient_for_second_moment=use_raw_gradient_for_second_moment,
            descent_check=descent_check,
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_history_variables(var_list)
        self._m, self._u = [], []
        for var in var_list:
            self._m.append(self.add_variable_from_reference(var, name="m"))
            self._u.append(self.add_variable_from_reference(var, name="u"))

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, dtype)
        b1 = tf.cast(self.beta_1, dtype)
        b2 = tf.cast(self.beta_2, dtype)
        eps_t = tf.cast(self.epsilon, dtype)
        step = tf.cast(self.iterations + 1, dtype)

        _, frac_grad, second_grad = self._direction_and_second_moment_grads(
            gradient, variable
        )

        m = self._m[idx]
        u = self._u[idx]

        new_m = b1 * _read(m) + (1.0 - b1) * frac_grad
        new_u = tf.maximum(b2 * _read(u), tf.abs(second_grad))

        self.assign(m, new_m)
        self.assign(u, new_u)

        b1_pow = tf.pow(b1, step)
        self.assign_sub(variable, lr_t / (1.0 - b1_pow) * new_m / (new_u + eps_t))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# Export list
# =============================================================================

__all__ = [
    "MemoryFSGD",
    "MemoryFRMSprop",
    "MemoryFAdam",
    "MemoryFAdadelta",
    "MemoryFAdagrad",
    "MemoryFAdamW",
    "MemoryFNadam",
    "MemoryFAdamax",
]


# =============================================================================
# Smoke tests
# =============================================================================

if __name__ == "__main__":
    print("Smoke tests for class_c_tf_fractional_optimizers_SR_keras3.py")
    print("=" * 68)

    keras.utils.set_random_seed(42)

    x = tf.random.normal((128, 10))
    y = tf.random.uniform((128,), minval=0, maxval=3, dtype=tf.int32)

    configs = [
        ("MemoryFSGD nu=0.9", MemoryFSGD(learning_rate=1e-2, vderiv=0.9, history_size=6)),
        ("MemoryFSGD nu=1.5", MemoryFSGD(learning_rate=1e-2, vderiv=1.5, history_size=6)),
        ("MemoryFRMSprop", MemoryFRMSprop(learning_rate=1e-3, vderiv=0.9, history_size=6)),
        ("MemoryFAdam", MemoryFAdam(learning_rate=1e-3, vderiv=0.9, history_size=6)),
        ("MemoryFAdadelta", MemoryFAdadelta(learning_rate=1.0, vderiv=0.9, history_size=6)),
        ("MemoryFAdagrad", MemoryFAdagrad(learning_rate=1e-2, vderiv=0.9, history_size=6)),
        ("MemoryFAdamW", MemoryFAdamW(learning_rate=1e-3, vderiv=0.9, history_size=6,
                                      weight_decay=0.004)),
        ("MemoryFNadam", MemoryFNadam(learning_rate=1e-3, vderiv=0.9, history_size=6)),
        ("MemoryFAdamax", MemoryFAdamax(learning_rate=1e-3, vderiv=0.9, history_size=6)),
        ("MemoryFAdam nu=1 (classical fallback)",
         MemoryFAdam(learning_rate=1e-3, vderiv=1.0, history_size=6)),
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
        hist = model.fit(x, y, epochs=12, batch_size=32, verbose=0)
        losses = hist.history["loss"]
        print(
            f"  {name:38s} loss {losses[0]:.4f} -> {losses[-1]:.4f}"
            f"  (DC gain={opt.gl_coefficient_sum():+.4f})"
        )
        assert losses[-1] < losses[0], f"{name}: loss did not decrease"

    print("=" * 68)
    print("All smoke tests passed.")
