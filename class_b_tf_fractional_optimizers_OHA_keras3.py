"""
class_b_tf_fractional_optimizers_OHA_keras3.py

Herrera-style fractional optimizers for TensorFlow >= 2.16 / Keras 3.

This file is the Keras 3 version of `class_b_tf_fractional_optimizers_OHA.py`.
The update rules are the same, operation for operation. Only the plumbing
changes, from the old `optimizer_v2` API (`_set_hyper`, `_create_slots`,
`_resource_apply_dense`) to the Keras 3 optimizer API:

    legacy optimizer_v2                 Keras 3
    ---------------------------------   -----------------------------------
    _set_hyper("learning_rate", lr)     super().__init__(learning_rate=lr)
    _create_slots(var_list)             build(var_list)
    add_slot(var, "m")                  add_variable_from_reference(var, "m")
    _resource_apply_dense(grad, var)    update_step(gradient, variable, lr)
    _decayed_lr(dtype)                  `learning_rate` argument of update_step
    var.assign_sub(...)                 self.assign_sub(variable, ...)

Backend: TensorFlow (the Keras 3 default when Keras is installed together
with TensorFlow). The update steps use TensorFlow ops directly so that the
arithmetic matches the legacy implementation exactly.

Main idea
---------
These optimizers follow the construction introduced by Oscar Herrera-Alcántara
and collaborators for "fractional derivative gradient-based optimizers".

For a parameter tensor w and ordinary gradient g = dL/dw, the modified gradient is

    g^(nu) = g * f_nu(w),

with

    f_nu(w) = (|w| + epsilon)^(1 - nu) / Gamma(2 - nu),

where
    - nu is the fractional order (`vderiv` below),
    - epsilon > 0 prevents singularities at w = 0,
    - Gamma(.) is the Gamma function.

This means:
    - the base optimizer structure is preserved,
    - only the gradient input is modified,
    - the fractional effect is local in optimization time,
    - no additional fractional memory/history is introduced.

At nu = 1, f_nu(w) = 1 for every w and each optimizer reduces to its base
optimizer.

Implemented optimizers
----------------------
    FSGD        (SGD, optional momentum / Nesterov)
    FAdam       (Adam, optional AMSGrad)
    FAdagrad    (Adagrad)
    FAdadelta   (Adadelta)
    FRMSprop    (RMSprop, optional momentum / centered)
    FAdamW      (AdamW, decoupled weight decay applied after the Adam step)

Compatibility
-------------
- `learning_rate=` is the Keras 3 argument. `lr=` is still accepted as an
  alias so that old experiment scripts keep working.
- All Keras 3 base-optimizer arguments are available through **kwargs:
  clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum,
  ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps,
  and weight_decay (Keras-style decoupled decay, applied before the step).
- Learning-rate schedules (keras.optimizers.schedules.*) work as usual.
- Every class is registered with `keras.saving.register_keras_serializable`,
  so models compiled with these optimizers can be saved to `.keras` files and
  loaded again without passing `custom_objects` (as long as this module has
  been imported).
- Optimizer checkpoints written by the legacy file cannot be restored into
  these classes (the variable layout of Keras 3 optimizers is different).

References
----------
1. Herrera-Alcántara, O.
   Fractional Derivative Gradient-Based Optimizers for Neural Networks and
   Human Activity Recognition.
   Applied Sciences, 12(18), 9264, 2022.
   https://doi.org/10.3390/app12189264

2. Herrera-Alcántara, O., Castelán-Aguilar, J. R.
   Fractional Gradient Optimizers for PyTorch: Enhancing GAN and BERT.
   Fractal and Fractional, 7(7), 500, 2023.
   https://doi.org/10.3390/fractalfract7070500

Usage example
-------------
    import keras
    from class_b_tf_fractional_optimizers_OHA_keras3 import FAdam

    optimizer = FAdam(learning_rate=1e-3, vderiv=0.9, frac_epsilon=1e-6)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

Notes
-----
- Sparse gradients (tf.IndexedSlices, e.g. from Embedding layers) are
  densified, exactly as in the legacy file.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import keras
import tensorflow as tf

if keras.backend.backend() != "tensorflow":  # pragma: no cover
    raise ImportError(
        "class_b_tf_fractional_optimizers_OHA_keras3 requires the TensorFlow "
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


# =============================================================================
# Shared helper base
# =============================================================================


class _FractionalHerreraBase:
    """
    Shared helper functionality for Herrera-style fractional gradient scaling.
    """

    def _init_fractional_args(self, vderiv: float = 1.0, frac_epsilon: float = 1e-6) -> None:
        if not isinstance(vderiv, (float, int)):
            raise TypeError("`vderiv` must be a float.")
        if not (0.0 < float(vderiv) < 2.0):
            raise ValueError("`vderiv` must satisfy 0 < vderiv < 2.")
        if frac_epsilon <= 0.0:
            raise ValueError("`frac_epsilon` must be > 0.")

        self.vderiv = float(vderiv)
        self.frac_epsilon = float(frac_epsilon)
        self._gamma_factor = float(math.gamma(2.0 - self.vderiv))

    def _to_dense_gradient(self, grad, var):
        """
        Convert sparse gradients to dense tensors (same as the legacy file).
        """
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                indices=tf.expand_dims(grad.indices, axis=1),
                updates=grad.values,
                shape=tf.shape(var),
            )
        return grad

    def _fractional_factor(self, var):
        """
        Compute the Herrera fractional scaling factor for the current variable.
        """
        dtype = _dtype(var)
        eps = tf.cast(self.frac_epsilon, dtype)
        exponent = tf.cast(1.0 - self.vderiv, dtype)
        gamma_val = tf.cast(self._gamma_factor, dtype)

        return tf.pow(tf.abs(var) + eps, exponent) / gamma_val

    def _fractional_gradient(self, grad, var):
        """
        Apply the Herrera scaling factor to the current gradient.
        """
        grad = self._to_dense_gradient(grad, var)
        grad = tf.cast(grad, _dtype(var))
        return grad * self._fractional_factor(var)

    def _fractional_config(self) -> Dict[str, Any]:
        return {
            "vderiv": self.vderiv,
            "frac_epsilon": self.frac_epsilon,
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
# FSGD
# =============================================================================


@keras.saving.register_keras_serializable(package="HerreraFractional")
class FSGD(_FractionalHerreraBase, keras.optimizers.Optimizer):
    """
    Herrera-style fractional SGD.

    Update without momentum:
        g_t^(nu) = g_t * f_nu(w_t)
        w_{t+1} = w_t - lr * g_t^(nu)

    Update with momentum:
        m_t = momentum * m_{t-1} - lr * g_t^(nu)
        w_{t+1} = w_t + m_t
    (Nesterov: w_{t+1} = w_t + momentum * m_t - lr * g_t^(nu))
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lr: float = None,
        momentum: float = 0.0,
        nesterov: bool = False,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        name: str = "FSGD",
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
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        if self.momentum > 0.0:
            for var in var_list:
                self._momentums.append(
                    self.add_variable_from_reference(var, name="momentum")
                )

    def update_step(self, gradient, variable, learning_rate):
        var_dtype = _dtype(variable)
        lr_t = tf.cast(learning_rate, var_dtype)

        grad = self._fractional_gradient(gradient, variable)

        if self.momentum > 0.0:
            momentum_coeff = tf.cast(self.momentum, var_dtype)
            m = self._momentums[self._get_variable_index(variable)]

            new_m = momentum_coeff * _read(m) - lr_t * grad
            self.assign(m, new_m)

            if self.nesterov:
                self.assign_add(variable, momentum_coeff * new_m - lr_t * grad)
            else:
                self.assign_add(variable, new_m)
            return

        self.assign_sub(variable, lr_t * grad)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# FAdam
# =============================================================================


@keras.saving.register_keras_serializable(package="HerreraFractional")
class FAdam(_FractionalHerreraBase, keras.optimizers.Optimizer):
    """
    Herrera-style fractional Adam.

        g_t^(nu) = g_t * f_nu(w_t)
        m_t = beta_1 m_{t-1} + (1 - beta_1) g_t^(nu)
        v_t = beta_2 v_{t-1} + (1 - beta_2) (g_t^(nu))^2
        w_{t+1} = w_t - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)

    with m_hat_t = m_t / (1 - beta_1^t), v_hat_t = v_t / (1 - beta_2^t)
    (or the running maximum of v_t for AMSGrad).
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
        frac_epsilon: float = 1e-6,
        name: str = "FAdam",
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
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._m, self._v, self._vhat = [], [], []
        for var in var_list:
            self._m.append(self.add_variable_from_reference(var, name="m"))
            self._v.append(self.add_variable_from_reference(var, name="v"))
            if self.amsgrad:
                self._vhat.append(self.add_variable_from_reference(var, name="vhat"))

    def update_step(self, gradient, variable, learning_rate):
        var_dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, var_dtype)
        beta_1_t = tf.cast(self.beta_1, var_dtype)
        beta_2_t = tf.cast(self.beta_2, var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)
        step_t = tf.cast(self.iterations + 1, var_dtype)

        grad = self._fractional_gradient(gradient, variable)

        m = self._m[idx]
        v = self._v[idx]

        new_m = beta_1_t * _read(m) + (1.0 - beta_1_t) * grad
        new_v = beta_2_t * _read(v) + (1.0 - beta_2_t) * tf.square(grad)

        self.assign(m, new_m)
        self.assign(v, new_v)

        beta_1_power = tf.pow(beta_1_t, step_t)
        beta_2_power = tf.pow(beta_2_t, step_t)

        m_hat = new_m / (1.0 - beta_1_power)

        if self.amsgrad:
            vhat = self._vhat[idx]
            new_vhat = tf.maximum(_read(vhat), new_v)
            self.assign(vhat, new_vhat)
            v_hat = new_vhat / (1.0 - beta_2_power)
        else:
            v_hat = new_v / (1.0 - beta_2_power)

        self.assign_sub(variable, lr_t * m_hat / (tf.sqrt(v_hat) + eps_t))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# FAdagrad
# =============================================================================


@keras.saving.register_keras_serializable(package="HerreraFractional")
class FAdagrad(_FractionalHerreraBase, keras.optimizers.Optimizer):
    """
    Herrera-style fractional Adagrad.

        acc_t = acc_{t-1} + (g_t^(nu))^2
        w_{t+1} = w_t - lr * g_t^(nu) / (sqrt(acc_t) + epsilon)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        name: str = "FAdagrad",
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
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        initializer = keras.initializers.Constant(self.initial_accumulator_value)
        self._accumulators = [
            self.add_variable_from_reference(var, name="accumulator", initializer=initializer)
            for var in var_list
        ]

    def update_step(self, gradient, variable, learning_rate):
        var_dtype = _dtype(variable)
        lr_t = tf.cast(learning_rate, var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)

        grad = self._fractional_gradient(gradient, variable)
        acc = self._accumulators[self._get_variable_index(variable)]

        new_acc = _read(acc) + tf.square(grad)
        self.assign(acc, new_acc)

        self.assign_sub(variable, lr_t * grad / (tf.sqrt(new_acc) + eps_t))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_accumulator_value": self.initial_accumulator_value,
                "epsilon": self.epsilon,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# FAdadelta
# =============================================================================


@keras.saving.register_keras_serializable(package="HerreraFractional")
class FAdadelta(_FractionalHerreraBase, keras.optimizers.Optimizer):
    """
    Herrera-style fractional Adadelta.

        E[g^2]_t  = rho E[g^2]_{t-1} + (1 - rho) (g_t^(nu))^2
        delta_t   = - sqrt(E[dx^2]_{t-1} + eps) / sqrt(E[g^2]_t + eps) * g_t^(nu)
        E[dx^2]_t = rho E[dx^2]_{t-1} + (1 - rho) delta_t^2
        w_{t+1}   = w_t + lr * delta_t
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        lr: float = None,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        name: str = "FAdadelta",
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
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._accumulated_grads = []
        self._accumulated_deltas = []
        for var in var_list:
            self._accumulated_grads.append(
                self.add_variable_from_reference(var, name="accumulated_grad")
            )
            self._accumulated_deltas.append(
                self.add_variable_from_reference(var, name="accumulated_delta")
            )

    def update_step(self, gradient, variable, learning_rate):
        var_dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, var_dtype)
        rho_t = tf.cast(self.rho, var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)

        grad = self._fractional_gradient(gradient, variable)

        acc_g = self._accumulated_grads[idx]
        acc_d = self._accumulated_deltas[idx]
        acc_d_value = _read(acc_d)

        new_acc_g = rho_t * _read(acc_g) + (1.0 - rho_t) * tf.square(grad)
        self.assign(acc_g, new_acc_g)

        rms_delta = tf.sqrt(acc_d_value + eps_t)
        rms_grad = tf.sqrt(new_acc_g + eps_t)
        delta = - (rms_delta / rms_grad) * grad

        new_acc_d = rho_t * acc_d_value + (1.0 - rho_t) * tf.square(delta)
        self.assign(acc_d, new_acc_d)

        self.assign_add(variable, lr_t * delta)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rho": self.rho,
                "epsilon": self.epsilon,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# FRMSprop
# =============================================================================


@keras.saving.register_keras_serializable(package="HerreraFractional")
class FRMSprop(_FractionalHerreraBase, keras.optimizers.Optimizer):
    """
    Herrera-style fractional RMSprop.

        v_t = rho v_{t-1} + (1 - rho) (g_t^(nu))^2
        denom = sqrt(v_t + eps)                 (centered: sqrt(v_t - a_t^2 + eps))
        w_{t+1} = w_t - lr * g_t^(nu) / denom    (momentum: via a momentum buffer)
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
        frac_epsilon: float = 1e-6,
        name: str = "FRMSprop",
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
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._velocities, self._momentums, self._average_grads = [], [], []
        for var in var_list:
            self._velocities.append(self.add_variable_from_reference(var, name="velocity"))
            if self.momentum > 0.0:
                self._momentums.append(self.add_variable_from_reference(var, name="momentum"))
            if self.centered:
                self._average_grads.append(self.add_variable_from_reference(var, name="avg_grad"))

    def update_step(self, gradient, variable, learning_rate):
        var_dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, var_dtype)
        rho_t = tf.cast(self.rho, var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)

        grad = self._fractional_gradient(gradient, variable)

        velocity = self._velocities[idx]
        new_velocity = rho_t * _read(velocity) + (1.0 - rho_t) * tf.square(grad)
        self.assign(velocity, new_velocity)

        if self.centered:
            avg_grad = self._average_grads[idx]
            new_avg_grad = rho_t * _read(avg_grad) + (1.0 - rho_t) * grad
            self.assign(avg_grad, new_avg_grad)
            denom = tf.sqrt(new_velocity - tf.square(new_avg_grad) + eps_t)
        else:
            denom = tf.sqrt(new_velocity + eps_t)

        if self.momentum > 0.0:
            momentum_coeff = tf.cast(self.momentum, var_dtype)
            mom = self._momentums[idx]
            new_mom = momentum_coeff * _read(mom) + lr_t * grad / denom
            self.assign(mom, new_mom)
            self.assign_sub(variable, new_mom)
            return

        self.assign_sub(variable, lr_t * grad / denom)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# FAdamW
# =============================================================================


@keras.saving.register_keras_serializable(package="HerreraFractional")
class FAdamW(_FractionalHerreraBase, keras.optimizers.Optimizer):
    """
    Herrera-style fractional AdamW with decoupled weight decay.

    The weight decay is applied after the Adam step, on the updated weights,
    exactly as in the legacy implementation:

        w' = w_t - lr * m_hat_t / (sqrt(v_hat_t) + eps)
        w_{t+1} = w' - lr * weight_decay * w'

    (Keras' own AdamW applies its decay before the step. To keep the legacy
    ordering, the base-class decay hook is disabled for this class and the
    decay is applied inside `update_step`.)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        weight_decay: float = 0.004,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        name: str = "FAdamW",
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=_resolve_learning_rate(learning_rate, lr),
            name=name,
            **kwargs,
        )

        if weight_decay is None or weight_decay < 0.0:
            raise ValueError("`weight_decay` must be >= 0.")
        if not (0.0 <= beta_1 < 1.0):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError("`beta_2` must satisfy 0 <= beta_2 < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.weight_decay = float(weight_decay)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = bool(amsgrad)
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def _apply_weight_decay(self, variables):
        # Disabled on purpose: the decay is applied after the Adam step inside
        # update_step(), as in the legacy implementation.
        return

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._m, self._v, self._vhat = [], [], []
        for var in var_list:
            self._m.append(self.add_variable_from_reference(var, name="m"))
            self._v.append(self.add_variable_from_reference(var, name="v"))
            if self.amsgrad:
                self._vhat.append(self.add_variable_from_reference(var, name="vhat"))

    def update_step(self, gradient, variable, learning_rate):
        var_dtype = _dtype(variable)
        idx = self._get_variable_index(variable)
        lr_t = tf.cast(learning_rate, var_dtype)
        beta_1_t = tf.cast(self.beta_1, var_dtype)
        beta_2_t = tf.cast(self.beta_2, var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)
        wd_t = tf.cast(self.weight_decay, var_dtype)
        step_t = tf.cast(self.iterations + 1, var_dtype)

        grad = self._fractional_gradient(gradient, variable)

        m = self._m[idx]
        v = self._v[idx]

        new_m = beta_1_t * _read(m) + (1.0 - beta_1_t) * grad
        new_v = beta_2_t * _read(v) + (1.0 - beta_2_t) * tf.square(grad)

        self.assign(m, new_m)
        self.assign(v, new_v)

        beta_1_power = tf.pow(beta_1_t, step_t)
        beta_2_power = tf.pow(beta_2_t, step_t)

        m_hat = new_m / (1.0 - beta_1_power)

        if self.amsgrad:
            vhat = self._vhat[idx]
            new_vhat = tf.maximum(_read(vhat), new_v)
            self.assign(vhat, new_vhat)
            v_hat = new_vhat / (1.0 - beta_2_power)
        else:
            v_hat = new_v / (1.0 - beta_2_power)

        # Adam step
        self.assign_sub(variable, lr_t * m_hat / (tf.sqrt(v_hat) + eps_t))

        # Decoupled weight decay on the updated weights
        if self.weight_decay > 0.0 and self._use_weight_decay(variable):
            self.assign_sub(variable, lr_t * wd_t * _read(variable))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# Export list
# =============================================================================

__all__ = [
    "FSGD",
    "FAdam",
    "FAdagrad",
    "FAdadelta",
    "FRMSprop",
    "FAdamW",
]


# =============================================================================
# Minimal smoke test
# =============================================================================

if __name__ == "__main__":
    keras.utils.set_random_seed(42)

    x = tf.random.normal((128, 10))
    y = tf.random.uniform((128,), minval=0, maxval=3, dtype=tf.int32)

    configs = [
        ("FSGD", FSGD(learning_rate=1e-2, momentum=0.9, vderiv=0.9)),
        ("FAdam", FAdam(learning_rate=1e-3, vderiv=0.9)),
        ("FAdagrad", FAdagrad(learning_rate=1e-2, vderiv=0.9)),
        ("FAdadelta", FAdadelta(learning_rate=1.0, vderiv=0.9)),
        ("FRMSprop", FRMSprop(learning_rate=1e-3, vderiv=0.9)),
        ("FAdamW", FAdamW(learning_rate=1e-3, weight_decay=0.004, vderiv=0.9)),
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
        print(f"  {name:10s} loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        assert losses[-1] < losses[0], f"{name}: loss did not decrease"

    print("All smoke tests passed.")
