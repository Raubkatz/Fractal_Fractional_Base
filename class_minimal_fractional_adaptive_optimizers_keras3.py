"""
class_minimal_fractional_adaptive_optimizers_keras3.py

Adaptive memory-based fractional optimizers for TensorFlow >= 2.16 / Keras 3.

This file is the Keras 3 version of
`class_minimal_fractional_adaptive_optimizers.py` (the corrected and extended
version with circular history buffer, descent safeguard, scalar mix/stability
state and Keras-exact Adam). The update rules are the same, operation for
operation. Only the optimizer plumbing changes, from `optimizer_v2`
(`_set_hyper`, `_create_slots`, `_resource_apply_dense`) to the Keras 3 API
(`build`, `update_step`).

Backend: TensorFlow (the Keras 3 default when Keras is installed together
with TensorFlow). The update steps use TensorFlow ops directly so that the
arithmetic matches the legacy implementation exactly.

Main idea
---------
    1. Keep the fractional order nu fixed.
    2. Compute a short-memory fractional gradient
           g_t^(nu) = sum_{k=0}^{K-1} c_k^(nu) g_{t-k},
           c_0 = 1, c_k = (1 - (nu+1)/k) c_{k-1}.
    3. Descent safeguard (use g_t if <g_t^(nu), g_t> <= 0), then match the
       fractional-gradient norm to the raw-gradient norm:
           g_hat_t^(nu) = g_t^(nu) ||g_t|| / (||g_t^(nu)|| + eps).
    4. Mix raw and fractional gradients:
           g_eff = (1 - lambda_t) g_t + lambda_t g_hat_t^(nu).
    5. Adapt only lambda_t, not the fractional order:
           s_t      = min(||g_t - g_{t-1}|| / (||g_{t-1}|| + eps), clip)
           S_t      = gamma_s S_{t-1} + (1 - gamma_s) s_t        (stability EMA)
           target   = lambda_{t-1} + step   if S_t <= threshold   (stable)
                      lambda_{t-1} - step   otherwise
           target  -= loss_mix_decrease     if the smoothed loss got worse
           target   = clip(target, min_mix, max_mix_t)            (warm-up cap)
           lambda_t = clip(gamma_m lambda_{t-1} + (1 - gamma_m) target,
                           min_mix, max_mix_t)

lambda_t = 0 gives exactly the base optimizer; lambda_t > 0 adds
fractional-memory smoothing. At nu = 1, g_eff = g_t for every lambda_t.

Implemented optimizers
----------------------
    AdaptiveMemoryFSGD          (SGD, optional momentum / Nesterov)
    AdaptiveMemoryFRMSprop      (RMSprop, optional momentum / centered)
    AdaptiveMemoryFAdam         (Adam, optional AMSGrad)
    AdaptiveMemoryFAdadelta     (Adadelta)
    AdaptiveMemoryFAdagrad      (Adagrad)
    AdaptiveMemoryFAdamW        (AdamW, decoupled weight decay after the step)
    AdaptiveMemoryFNadam        (Nadam, Nesterov-Adam)
    AdaptiveMemoryFAdamax       (Adamax, infinity-norm second moment)

plus the optional callback `AdaptiveLossFeedback` (see below).

Recommended defaults (unchanged)
--------------------------------
Neural networks:  vderiv = 0.85-0.9, history_size = 3-4,
                  initial_mix = 0.10, max_mix = 0.30, adapt_step = 0.01,
                  stability_threshold = 0.25.
Difficult analytic surfaces: history_size = 4-6, max_mix = 0.40,
                  adapt_step = 0.015.

Optional loss feedback
----------------------
`set_current_loss(loss_value)` feeds the training loss to the optimizer; when
the smoothed loss becomes worse than the best smoothed loss so far, lambda_t
is pushed down by `loss_mix_decrease` in the following steps. Call it from
eager Python (a custom loop, or the `AdaptiveLossFeedback` callback for
`model.fit`). Without it (the setting used in the paper experiments) the
loss term is always zero.

One deliberate plumbing difference from the legacy file: the "loss got
worse" flag is now an optimizer variable instead of a Python float. In the
legacy file a tf.function (e.g. model.fit) froze the Python float at trace
time, so set_current_loss() had no effect on compiled training steps. Here
it does. When set_current_loss() is never called, or in eager mode, the
results are identical to the legacy file.

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
    from class_minimal_fractional_adaptive_optimizers_keras3 import (
        AdaptiveMemoryFAdam, AdaptiveLossFeedback)

    optimizer = AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=0.9, history_size=4)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
    model.fit(x, y, epochs=10)                                   # paper setting
    model.fit(x, y, epochs=10, callbacks=[AdaptiveLossFeedback()])  # optional
    print(optimizer.current_mix_values())                        # lambda_t per variable
"""

from __future__ import annotations

from typing import Any, Dict, List

import keras
import tensorflow as tf

if keras.backend.backend() != "tensorflow":  # pragma: no cover
    raise ImportError(
        "class_minimal_fractional_adaptive_optimizers_keras3 requires the "
        "TensorFlow backend of Keras 3 (KERAS_BACKEND=tensorflow). Current "
        f"backend: {keras.backend.backend()!r}."
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
# Shared adaptive memory-fractional helper base
# =============================================================================


class _AdaptiveMemoryFractionalBase:
    """
    Shared helper functionality for adaptive memory-based fractional
    optimizers with a circular gradient-history buffer.
    """

    _CLASSICAL_TOL = 1e-12

    # ------------------------------------------------------------------
    # Init and coefficients
    # ------------------------------------------------------------------

    def _init_adaptive_memory_fractional_args(
        self,
        vderiv: float = 0.9,
        history_size: int = 4,
        normalize_coefficients: bool = True,
        coeff_epsilon: float = 1e-12,
        initial_mix: float = 0.10,
        min_mix: float = 0.0,
        max_mix: float = 0.30,
        adapt_step: float = 0.01,
        mix_ema_gamma: float = 0.90,
        stability_ema_gamma: float = 0.95,
        stability_threshold: float = 0.25,
        stability_clip: float = 10.0,
        norm_match_fractional_gradient: bool = True,
        use_raw_gradient_for_second_moment: bool = True,
        loss_ema_beta: float = 0.95,
        loss_worsen_tolerance: float = 1e-4,
        loss_mix_decrease: float = 0.02,
        warmup_steps: int = 0,
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

        if not (0.0 <= min_mix <= initial_mix <= max_mix <= 1.0):
            raise ValueError(
                "Mix values must satisfy 0 <= min_mix <= initial_mix <= max_mix <= 1."
            )
        if adapt_step < 0.0:
            raise ValueError("`adapt_step` must be >= 0.")
        if not (0.0 <= mix_ema_gamma < 1.0):
            raise ValueError("`mix_ema_gamma` must satisfy 0 <= mix_ema_gamma < 1.")
        if not (0.0 <= stability_ema_gamma < 1.0):
            raise ValueError(
                "`stability_ema_gamma` must satisfy 0 <= stability_ema_gamma < 1."
            )
        if stability_threshold < 0.0:
            raise ValueError("`stability_threshold` must be >= 0.")
        if stability_clip <= 0.0:
            raise ValueError("`stability_clip` must be > 0.")
        if not (0.0 <= loss_ema_beta < 1.0):
            raise ValueError("`loss_ema_beta` must satisfy 0 <= loss_ema_beta < 1.")
        if loss_worsen_tolerance < 0.0:
            raise ValueError("`loss_worsen_tolerance` must be >= 0.")
        if loss_mix_decrease < 0.0:
            raise ValueError("`loss_mix_decrease` must be >= 0.")
        if warmup_steps < 0:
            raise ValueError("`warmup_steps` must be >= 0.")

        self.vderiv = float(vderiv)
        self.history_size = int(history_size)
        self.normalize_coefficients = bool(normalize_coefficients)
        self.coeff_epsilon = float(coeff_epsilon)

        self.initial_mix = float(initial_mix)
        self.min_mix = float(min_mix)
        self.max_mix = float(max_mix)
        self.adapt_step = float(adapt_step)
        self.mix_ema_gamma = float(mix_ema_gamma)
        self.stability_ema_gamma = float(stability_ema_gamma)
        self.stability_threshold = float(stability_threshold)
        self.stability_clip = float(stability_clip)

        self.norm_match_fractional_gradient = bool(norm_match_fractional_gradient)
        self.use_raw_gradient_for_second_moment = bool(
            use_raw_gradient_for_second_moment
        )

        self.loss_ema_beta = float(loss_ema_beta)
        self.loss_worsen_tolerance = float(loss_worsen_tolerance)
        self.loss_mix_decrease = float(loss_mix_decrease)
        self.warmup_steps = int(warmup_steps)
        self.descent_check = bool(descent_check)

        self._fractional_coefficients = self._compute_fractional_coefficients(
            self.vderiv,
            self.history_size,
            self.normalize_coefficients,
        )

        self._current_loss_value = None
        self._loss_ema_value = None
        self._best_loss_ema_value = None
        self._loss_worse_value = 0.0
        self._loss_worse_flag = None  # optimizer variable, created in build()

    def _compute_fractional_coefficients(
        self,
        nu: float,
        n_terms: int,
        normalize: bool = True,
    ) -> List[float]:
        """
        Short-memory Grunwald-Letnikov coefficients.

        c_0 = 1
        c_k = (1 - (nu + 1)/k) c_{k-1},  k >= 1

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
        """Diagnostic: DC gain of the (possibly normalized) GL kernel."""
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
    # Tensor helpers
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
    # Optimizer variables
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

    def _build_adaptive_memory_variables(self, var_list) -> None:
        self._grad_histories = []
        self._prev_raw_grads = []
        self._frac_mixes = []
        self._stability_states = []
        for var in var_list:
            if self._is_classical:
                self._grad_histories.append(None)
            else:
                self._grad_histories.append(
                    self._add_shaped_variable(
                        var, (self.history_size,) + tuple(var.shape), "grad_history"
                    )
                )
            self._prev_raw_grads.append(
                self.add_variable_from_reference(var, name="prev_raw_grad")
            )
            self._frac_mixes.append(
                self._add_shaped_variable(
                    var, (), "frac_mix",
                    initializer=keras.initializers.Constant(self.initial_mix),
                )
            )
            self._stability_states.append(
                self._add_shaped_variable(var, (), "stability_state")
            )
        self._loss_worse_flag = self.add_variable(
            shape=(),
            initializer=keras.initializers.Constant(self._loss_worse_value),
            dtype="float32",
            name="loss_worse_flag",
        )

    # ------------------------------------------------------------------
    # Loss interface
    # ------------------------------------------------------------------

    def set_current_loss(self, loss_value):
        """
        Optional loss injection from eager Python (custom loop or callback).

        The optimizer does not require this, but if the loop calls it, the
        fractional mix is reduced when the smoothed loss becomes worse than
        the best smoothed loss observed so far.
        """
        try:
            loss_scalar = float(
                keras.ops.convert_to_numpy(tf.cast(loss_value, tf.float32))
            )
        except Exception:
            loss_scalar = float(loss_value)

        self._current_loss_value = loss_scalar

        if self._loss_ema_value is None:
            self._loss_ema_value = loss_scalar
            self._best_loss_ema_value = loss_scalar
            self._loss_worse_value = 0.0
            self._sync_loss_flag()
            return

        self._loss_ema_value = (
            self.loss_ema_beta * self._loss_ema_value
            + (1.0 - self.loss_ema_beta) * loss_scalar
        )

        threshold = self.loss_worsen_tolerance * max(
            1.0, abs(self._best_loss_ema_value)
        )

        if self._loss_ema_value < self._best_loss_ema_value - threshold:
            self._best_loss_ema_value = self._loss_ema_value
            self._loss_worse_value = 0.0
        elif self._loss_ema_value > self._best_loss_ema_value + threshold:
            self._loss_worse_value = 1.0
        else:
            self._loss_worse_value = 0.0
        self._sync_loss_flag()

    def _sync_loss_flag(self):
        if self._loss_worse_flag is not None:
            self._loss_worse_flag.assign(self._loss_worse_value)

    # ------------------------------------------------------------------
    # Fractional gradient (circular buffer) and adaptive mix
    # ------------------------------------------------------------------

    def _fractional_gradient(self, raw_grad, var):
        """
        Write g_t into the circular buffer and return the fractional
        gradient after the descent safeguard and norm matching.
        """
        dtype = _dtype(var)

        if self._is_classical:
            return raw_grad

        k = self.history_size
        hist_var = self._grad_histories[self._get_variable_index(var)]

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
        # Read after the single-row write (TensorFlow orders the resource ops).
        hist = _read(hist_var)

        coeffs = tf.constant(self._fractional_coefficients, dtype=dtype)
        lags = tf.math.floormod(pos - tf.range(k, dtype=tf.int32), k)
        c_perm = tf.gather(coeffs, lags)
        c_shape = [k] + [1] * len(var.shape)
        frac_grad = tf.reduce_sum(
            hist * tf.reshape(c_perm, c_shape), axis=0
        )

        if self.descent_check:
            dot = tf.reduce_sum(frac_grad * raw_grad)
            keep = tf.cast(dot > 0.0, dtype)
            frac_grad = keep * frac_grad + (1.0 - keep) * raw_grad

        if self.norm_match_fractional_gradient:
            raw_norm = self._global_l2_norm(raw_grad, dtype)
            frac_norm = self._global_l2_norm(frac_grad, dtype)
            scale = raw_norm / (frac_norm + tf.cast(self.coeff_epsilon, dtype))
            frac_grad = frac_grad * scale

        return frac_grad

    def _stability_score(self, raw_grad, prev_raw_grad, dtype):
        diff_norm = self._global_l2_norm(raw_grad - prev_raw_grad, dtype)
        prev_norm = self._global_l2_norm(prev_raw_grad, dtype)
        score = diff_norm / (prev_norm + tf.cast(self.coeff_epsilon, dtype))
        return tf.minimum(score, tf.cast(self.stability_clip, dtype))

    def _warmup_max_mix(self, dtype):
        if self.warmup_steps <= 0:
            return tf.cast(self.max_mix, dtype)
        step = tf.cast(self.iterations + 1, dtype)
        warmup = tf.cast(self.warmup_steps, dtype)
        fraction = tf.clip_by_value(step / warmup, 0.0, 1.0)
        return tf.cast(self.max_mix, dtype) * fraction

    def _adaptive_effective_gradient(self, grad, var):
        """
        Main shared adaptive step.

        Returns:
            raw_grad
            eff_grad
            second_moment_grad
        """
        dtype = _dtype(var)
        idx = self._get_variable_index(var)

        raw_grad = tf.cast(self._to_dense_gradient(grad, var), dtype)
        prev_raw_grad_var = self._prev_raw_grads[idx]
        mix_state = self._frac_mixes[idx]
        stability_state = self._stability_states[idx]

        prev_raw_grad = _read(prev_raw_grad_var)
        mix_scalar = tf.cast(_read(mix_state), dtype)
        stability_scalar = tf.cast(_read(stability_state), dtype)

        frac_grad = self._fractional_gradient(raw_grad, var)

        score = self._stability_score(raw_grad, prev_raw_grad, dtype)

        gamma_stab = tf.cast(self.stability_ema_gamma, dtype)
        new_stability = gamma_stab * stability_scalar + (1.0 - gamma_stab) * score

        stable = tf.cast(
            new_stability <= tf.cast(self.stability_threshold, dtype), dtype
        )

        step = tf.cast(self.adapt_step, dtype)
        target_mix = tf.where(
            stable > 0.5,
            mix_scalar + step,
            mix_scalar - step,
        )

        loss_worse = tf.cast(_read(self._loss_worse_flag), dtype)
        target_mix = target_mix - loss_worse * tf.cast(self.loss_mix_decrease, dtype)

        max_mix_t = self._warmup_max_mix(dtype)
        min_mix_t = tf.cast(self.min_mix, dtype)
        target_mix = tf.clip_by_value(target_mix, min_mix_t, max_mix_t)

        gamma_mix = tf.cast(self.mix_ema_gamma, dtype)
        new_mix = gamma_mix * mix_scalar + (1.0 - gamma_mix) * target_mix
        new_mix = tf.clip_by_value(new_mix, min_mix_t, max_mix_t)

        eff_grad = (1.0 - new_mix) * raw_grad + new_mix * frac_grad

        if self.use_raw_gradient_for_second_moment:
            second_moment_grad = raw_grad
        else:
            second_moment_grad = eff_grad

        self.assign(prev_raw_grad_var, raw_grad)
        self.assign(mix_state, tf.cast(new_mix, mix_state.dtype))
        self.assign(stability_state, tf.cast(new_stability, stability_state.dtype))

        return raw_grad, eff_grad, second_moment_grad

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def current_mix_values(self) -> List[tf.Tensor]:
        """Current mixing coefficient lambda_t, one scalar per model variable."""
        return [tf.reduce_mean(_read(w)) for w in getattr(self, "_frac_mixes", [])]

    def current_stability_values(self) -> List[tf.Tensor]:
        """Current smoothed stability score S_t, one scalar per model variable."""
        return [tf.reduce_mean(_read(w)) for w in getattr(self, "_stability_states", [])]

    def _adaptive_memory_fractional_config(self) -> Dict[str, Any]:
        return {
            "vderiv": self.vderiv,
            "history_size": self.history_size,
            "normalize_coefficients": self.normalize_coefficients,
            "coeff_epsilon": self.coeff_epsilon,
            "initial_mix": self.initial_mix,
            "min_mix": self.min_mix,
            "max_mix": self.max_mix,
            "adapt_step": self.adapt_step,
            "mix_ema_gamma": self.mix_ema_gamma,
            "stability_ema_gamma": self.stability_ema_gamma,
            "stability_threshold": self.stability_threshold,
            "stability_clip": self.stability_clip,
            "norm_match_fractional_gradient":
                self.norm_match_fractional_gradient,
            "use_raw_gradient_for_second_moment":
                self.use_raw_gradient_for_second_moment,
            "loss_ema_beta": self.loss_ema_beta,
            "loss_worsen_tolerance": self.loss_worsen_tolerance,
            "loss_mix_decrease": self.loss_mix_decrease,
            "warmup_steps": self.warmup_steps,
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


# Shared keyword plumbing: every optimizer below forwards these to
# _init_adaptive_memory_fractional_args.
_ADAPTIVE_KWARG_NAMES = (
    "vderiv", "history_size", "normalize_coefficients", "coeff_epsilon",
    "initial_mix", "min_mix", "max_mix", "adapt_step", "mix_ema_gamma",
    "stability_ema_gamma", "stability_threshold", "stability_clip",
    "norm_match_fractional_gradient", "use_raw_gradient_for_second_moment",
    "loss_ema_beta", "loss_worsen_tolerance", "loss_mix_decrease",
    "warmup_steps", "descent_check",
)


def _split_adaptive_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pop the adaptive-memory kwargs out of a kwargs dict."""
    return {k: kwargs.pop(k) for k in list(kwargs) if k in _ADAPTIVE_KWARG_NAMES}


# =============================================================================
# AdaptiveMemoryFSGD
# =============================================================================


@keras.saving.register_keras_serializable(package="AdaptiveMemoryFractional")
class AdaptiveMemoryFSGD(_AdaptiveMemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Adaptive memory-fractional SGD.

    Uses adaptive fractional mixing in the SGD update direction.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lr: float = None,
        momentum: float = 0.0,
        nesterov: bool = False,
        name: str = "AdaptiveMemoryFSGD",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
        super().__init__(
            learning_rate=_resolve_learning_rate(learning_rate, lr),
            name=name,
            **kwargs,
        )

        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")

        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_adaptive_memory_variables(var_list)
        self._momentums = []
        if self.momentum > 0.0:
            for var in var_list:
                self._momentums.append(
                    self.add_variable_from_reference(var, name="momentum")
                )

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        lr_t = tf.cast(learning_rate, dtype)

        _, eff_grad, _ = self._adaptive_effective_gradient(gradient, variable)

        if self.momentum > 0.0:
            mu = tf.cast(self.momentum, dtype)
            m = self._momentums[self._get_variable_index(variable)]
            new_m = mu * _read(m) - lr_t * eff_grad
            self.assign(m, new_m)
            if self.nesterov:
                self.assign_add(variable, mu * new_m - lr_t * eff_grad)
            else:
                self.assign_add(variable, new_m)
            return

        self.assign_sub(variable, lr_t * eff_grad)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFRMSprop
# =============================================================================


@keras.saving.register_keras_serializable(package="AdaptiveMemoryFractional")
class AdaptiveMemoryFRMSprop(_AdaptiveMemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Adaptive memory-fractional RMSprop.

    Adaptive fractional mixing in the numerator; the squared-gradient
    accumulator uses the raw gradient by default.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        centered: bool = False,
        name: str = "AdaptiveMemoryFRMSprop",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
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

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_adaptive_memory_variables(var_list)
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

        _, eff_grad, second_grad = self._adaptive_effective_gradient(gradient, variable)

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
            new_mom = mu * _read(mom) + lr_t * eff_grad / denom
            self.assign(mom, new_mom)
            self.assign_sub(variable, new_mom)
            return

        self.assign_sub(variable, lr_t * eff_grad / denom)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFAdam
# =============================================================================


@keras.saving.register_keras_serializable(package="AdaptiveMemoryFractional")
class AdaptiveMemoryFAdam(_AdaptiveMemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Adaptive memory-fractional Adam.

    Adaptive fractional mixing in the first moment; the second moment uses
    the raw gradient by default. Keras-exact Adam formulation (bias
    correction folded into the learning rate, epsilon outside the
    uncorrected sqrt), so at nu = 1 the update equals the standard Adam step.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        name: str = "AdaptiveMemoryFAdam",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
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

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_adaptive_memory_variables(var_list)
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

        _, eff_grad, second_grad = self._adaptive_effective_gradient(gradient, variable)

        m = self._m[idx]
        v = self._v[idx]

        new_m = b1 * _read(m) + (1.0 - b1) * eff_grad
        new_v = b2 * _read(v) + (1.0 - b2) * tf.square(second_grad)

        self.assign(m, new_m)
        self.assign(v, new_v)

        # Keras-exact formulation (bias correction folded into lr).
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
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFAdadelta
# =============================================================================


@keras.saving.register_keras_serializable(package="AdaptiveMemoryFractional")
class AdaptiveMemoryFAdadelta(_AdaptiveMemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Adaptive memory-fractional Adadelta.

    Adaptive fractional mixing in the update direction; the accumulators
    use the raw gradient / raw-driven update by default.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        lr: float = None,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        name: str = "AdaptiveMemoryFAdadelta",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
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

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_adaptive_memory_variables(var_list)
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

        _, eff_grad, second_grad = self._adaptive_effective_gradient(gradient, variable)

        acc_g = self._accumulated_grads[idx]
        acc_d = self._accumulated_deltas[idx]
        acc_d_value = _read(acc_d)

        new_acc_g = rho_t * _read(acc_g) + (1.0 - rho_t) * tf.square(second_grad)
        self.assign(acc_g, new_acc_g)

        rms_delta = tf.sqrt(acc_d_value + eps_t)
        rms_grad = tf.sqrt(new_acc_g + eps_t)
        delta = -(rms_delta / rms_grad) * eff_grad

        new_acc_d = rho_t * acc_d_value + (1.0 - rho_t) * tf.square(delta)
        self.assign(acc_d, new_acc_d)

        self.assign_add(variable, lr_t * delta)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rho": self.rho,
                "epsilon": self.epsilon,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFAdagrad
# =============================================================================


@keras.saving.register_keras_serializable(package="AdaptiveMemoryFractional")
class AdaptiveMemoryFAdagrad(_AdaptiveMemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Adaptive memory-fractional Adagrad.

    Update:
        acc_t = acc_{t-1} + s_t^2      (s_t = raw gradient by default)
        theta_{t+1} = theta_t - lr * g_eff_t / (sqrt(acc_t) + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-7,
        name: str = "AdaptiveMemoryFAdagrad",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
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

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_adaptive_memory_variables(var_list)
        init = keras.initializers.Constant(self.initial_accumulator_value)
        self._accumulators = [
            self.add_variable_from_reference(var, name="accumulator", initializer=init)
            for var in var_list
        ]

    def update_step(self, gradient, variable, learning_rate):
        dtype = _dtype(variable)
        lr_t = tf.cast(learning_rate, dtype)
        eps_t = tf.cast(self.epsilon, dtype)

        _, eff_grad, second_grad = self._adaptive_effective_gradient(gradient, variable)

        acc = self._accumulators[self._get_variable_index(variable)]
        new_acc = _read(acc) + tf.square(second_grad)
        self.assign(acc, new_acc)

        self.assign_sub(variable, lr_t * eff_grad / (tf.sqrt(new_acc) + eps_t))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_accumulator_value": self.initial_accumulator_value,
                "epsilon": self.epsilon,
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFAdamW
# =============================================================================


@keras.saving.register_keras_serializable(package="AdaptiveMemoryFractional")
class AdaptiveMemoryFAdamW(AdaptiveMemoryFAdam):
    """
    Adaptive memory-fractional AdamW: AdaptiveMemoryFAdam plus decoupled
    weight decay applied after the Adam step, on the updated weights,
        theta' = theta_t - lr * (Adam step)
        theta_{t+1} = theta' - lr * weight_decay * theta'.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        weight_decay: float = 0.004,
        name: str = "AdaptiveMemoryFAdamW",
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
# AdaptiveMemoryFNadam
# =============================================================================


@keras.saving.register_keras_serializable(package="AdaptiveMemoryFractional")
class AdaptiveMemoryFNadam(_AdaptiveMemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Adaptive memory-fractional Nadam (Nesterov-accelerated Adam).

    Update (Dozat's formulation without the momentum-product schedule):
        m_t   = beta_1 m_{t-1} + (1 - beta_1) g_eff_t
        v_t   = beta_2 v_{t-1} + (1 - beta_2) s_t^2
        m_bar = beta_1 m_t / (1 - beta_1^{t+1})
                + (1 - beta_1) g_eff_t / (1 - beta_1^t)
        theta_{t+1} = theta_t - lr * m_bar / (sqrt(v_t/(1-beta_2^t)) + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        name: str = "AdaptiveMemoryFNadam",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
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

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_adaptive_memory_variables(var_list)
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

        _, eff_grad, second_grad = self._adaptive_effective_gradient(gradient, variable)

        m = self._m[idx]
        v = self._v[idx]

        new_m = b1 * _read(m) + (1.0 - b1) * eff_grad
        new_v = b2 * _read(v) + (1.0 - b2) * tf.square(second_grad)

        self.assign(m, new_m)
        self.assign(v, new_v)

        b1_pow = tf.pow(b1, step)
        b2_pow = tf.pow(b2, step)

        m_bar = (
            b1 * new_m / (1.0 - b1_pow * b1)
            + (1.0 - b1) * eff_grad / (1.0 - b1_pow)
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
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# AdaptiveMemoryFAdamax
# =============================================================================


@keras.saving.register_keras_serializable(package="AdaptiveMemoryFractional")
class AdaptiveMemoryFAdamax(_AdaptiveMemoryFractionalBase, keras.optimizers.Optimizer):
    """
    Adaptive memory-fractional Adamax (infinity-norm Adam).

    Update:
        m_t = beta_1 m_{t-1} + (1 - beta_1) g_eff_t
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
        name: str = "AdaptiveMemoryFAdamax",
        **kwargs: Any,
    ):
        adaptive_kwargs = _split_adaptive_kwargs(kwargs)
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

        self._init_adaptive_memory_fractional_args(**adaptive_kwargs)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._build_adaptive_memory_variables(var_list)
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

        _, eff_grad, second_grad = self._adaptive_effective_gradient(gradient, variable)

        m = self._m[idx]
        u = self._u[idx]

        new_m = b1 * _read(m) + (1.0 - b1) * eff_grad
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
                **self._adaptive_memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# Optional callback: feed the training loss to the adaptive optimizers
# =============================================================================


class AdaptiveLossFeedback(keras.callbacks.Callback):
    """
    Calls `model.optimizer.set_current_loss(logs["loss"])` after every
    training batch, so the loss-aware decrease of lambda_t is active in
    `model.fit`. Note that Keras reports the running mean of the loss over the
    current epoch in `logs["loss"]`, so the optimizer sees an epoch-smoothed
    loss (on top of its own EMA with `loss_ema_beta`).
    """

    def on_train_batch_end(self, batch, logs=None):
        optimizer = getattr(self.model, "optimizer", None)
        if logs and "loss" in logs and hasattr(optimizer, "set_current_loss"):
            optimizer.set_current_loss(logs["loss"])


# =============================================================================
# Export list
# =============================================================================

__all__ = [
    "AdaptiveMemoryFSGD",
    "AdaptiveMemoryFRMSprop",
    "AdaptiveMemoryFAdam",
    "AdaptiveMemoryFAdadelta",
    "AdaptiveMemoryFAdagrad",
    "AdaptiveMemoryFAdamW",
    "AdaptiveMemoryFNadam",
    "AdaptiveMemoryFAdamax",
    "AdaptiveLossFeedback",
]


# =============================================================================
# Smoke tests
# =============================================================================

if __name__ == "__main__":
    print("Smoke tests for class_minimal_fractional_adaptive_optimizers_keras3.py")
    print("=" * 74)

    keras.utils.set_random_seed(42)

    x = tf.random.normal((128, 10))
    y = tf.random.uniform((128,), minval=0, maxval=3, dtype=tf.int32)

    common = dict(vderiv=0.9, history_size=4, initial_mix=0.10,
                  max_mix=0.30, adapt_step=0.01)

    configs = [
        ("AdaptiveMemoryFSGD", AdaptiveMemoryFSGD(learning_rate=1e-2, **common)),
        ("AdaptiveMemoryFSGD nu=1.5",
         AdaptiveMemoryFSGD(learning_rate=1e-2, vderiv=1.5, history_size=4,
                            initial_mix=0.10, max_mix=0.30, adapt_step=0.01)),
        ("AdaptiveMemoryFRMSprop", AdaptiveMemoryFRMSprop(learning_rate=1e-3, **common)),
        ("AdaptiveMemoryFAdam", AdaptiveMemoryFAdam(learning_rate=1e-3, **common)),
        ("AdaptiveMemoryFAdadelta", AdaptiveMemoryFAdadelta(learning_rate=1.0, **common)),
        ("AdaptiveMemoryFAdagrad", AdaptiveMemoryFAdagrad(learning_rate=1e-2, **common)),
        ("AdaptiveMemoryFAdamW",
         AdaptiveMemoryFAdamW(learning_rate=1e-3, weight_decay=0.004, **common)),
        ("AdaptiveMemoryFNadam", AdaptiveMemoryFNadam(learning_rate=1e-3, **common)),
        ("AdaptiveMemoryFAdamax", AdaptiveMemoryFAdamax(learning_rate=1e-3, **common)),
        ("AdaptiveMemoryFAdam nu=1 (classical)",
         AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=1.0, history_size=4)),
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
        mixes = opt.current_mix_values()
        mean_mix = float(tf.reduce_mean(tf.stack(mixes))) if mixes else float("nan")
        print(
            f"  {name:38s} loss {losses[0]:.4f} -> {losses[-1]:.4f}"
            f"  (mean lambda={mean_mix:.3f})"
        )
        assert losses[-1] < losses[0], f"{name}: loss did not decrease"

    print("=" * 74)
    print("All smoke tests passed.")
