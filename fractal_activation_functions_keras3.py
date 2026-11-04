"""
fractal_activation_functions_keras3.py

Fractal activation functions for TensorFlow >= 2.16 / Keras 3.

This file is the Keras 3 version of `fractal_activation_functions.py`. Every
function keeps its name, its signature, its default parameters and its
arithmetic (same TensorFlow ops in the same order, so values and gradients
are bit-identical to the original). What changed:

- `tensorflow.experimental.numpy` (tnp) is no longer needed: tnp.pi -> math.pi
  (the same float), tnp.exp/abs/tanh/sin -> the tf.math kernels that tnp
  dispatched to anyway.
- The unused matplotlib import and the duplicate imports are gone.
- Every function is registered with `keras.saving.register_keras_serializable`
  (package "FractalActivations"), so a model that uses them can be saved to a
  `.keras` file and loaded again (after importing this module).
- The functions that compute internally in float64 return the dtype of their
  input instead of a hard-coded float32. For the usual float32 networks this
  is the same result; under mixed precision the output now matches the layer.
- New helpers: dictionaries of all activations, `get_fractal_activation()`,
  and a `FractalActivation` layer for using an activation with non-default
  parameters in a model that can still be saved.

Usage
-----
    import keras
    import fractal_activation_functions_keras3 as faf

    # 1) as the activation of a Dense layer (default parameters)
    keras.layers.Dense(32, activation=faf.modified_weierstrass_function_tanh)

    # 2) as a separate layer, with non-default parameters, still saveable
    keras.layers.Dense(32)
    faf.FractalActivation("modified_weierstrass_function_tanh", n_terms=50)

Sources
-------
The fractal activation functions were introduced in
    Raubitzek, S.; Kietreiber, T.; Eresheim, S.; Buchelt, A.; Mallinger, K.
    Fractals in neural networks: Introducing a new class of activation
    functions. Commun. Nonlinear Sci. Numer. Simul. 161, 110064 (2026).
    https://doi.org/10.1016/j.cnsns.2026.110064
The Cantor-based activations follow
    Chipera, R.; Du, J.; Tsapara, I. Fractal and Chaotic Activation Functions
    in Echo State Networks: Preprocessing Topology Governs the Echo State
    Property. arXiv:2512.14675 (2025). https://arxiv.org/abs/2512.14675
The random_depth_* variants are extensions in this code base.

Activations used in the article
-------------------------------
    modulated_blancmange_curve
    decaying_cosine_function_tf
    modified_weierstrass_function_tanh
    weierstrass_mandelbrot_function_xpsin
(see PAPER_FRACTAL_ACTIVATIONS below). The remaining functions are kept for
completeness and for further experiments.

Numerical notes (unchanged behaviour of the original functions)
---------------------------------------------------------------
- modulated_blancmange_curve contains sqrt(|x|), whose derivative is
  unbounded at x = 0; the random-depth variant adds 1e-8 inside the sqrt.
- cantor_function_tf and cantor_set_activation_tf are piecewise constant at
  every finite depth (zero derivative almost everywhere), so gradients do
  not pass through them: hidden layers using them do not learn.
- Python integer powers such as 2**n and lambda_val**k are evaluated in
  Python and then multiplied into the tensor, exactly as before.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import keras
import tensorflow as tf

if keras.backend.backend() != "tensorflow":  # pragma: no cover
    raise ImportError(
        "fractal_activation_functions_keras3 requires the TensorFlow backend "
        "of Keras 3 (KERAS_BACKEND=tensorflow). Current backend: "
        f"{keras.backend.backend()!r}."
    )

_PI = math.pi  # identical to tnp.pi / np.pi


def _register(fn):
    return keras.saving.register_keras_serializable(package="FractalActivations")(fn)


def _output_dtype(x):
    """Float dtype of the input (falls back to float32 for non-float inputs)."""
    dtype = tf.as_dtype(getattr(x, "dtype", tf.float32))
    return dtype if dtype.is_floating else tf.float32


# ============================================================
# Fixed-depth fractal activation functions
# ============================================================


# Custom activation function: Revised Modulated Blancmange Curve
@_register
def modulated_blancmange_curve(x, n_terms=30, a=0.75):
    """
    y(x) = 1/2 * sum_{n=0}^{n_terms-1} tanh(a 2^n x) * | (2^n x mod 2) - a sqrt|x| | / 2^n
    """
    y = tf.zeros_like(x)
    for n in range(n_terms):
        factor = 2 ** n
        modulation = tf.tanh(a * factor * x)  # Modulating factor using tanh
        ax = a * tf.sqrt(tf.abs(x))
        y += modulation * tf.abs(x * factor % 2 - 1 * ax) / factor
    return y / 2


def _mirrored_function_tf(x):
    """+1 for x < 0, -1 otherwise (used by the decaying cosine functions)."""
    return tf.where(x < 0, tf.ones_like(x), -tf.ones_like(x))


# Blended function with tanh and cosine components with exponential decay
@_register
def decaying_cosine_function_tf(x, a=0.5, b=3, c=0.5, d=2, n_terms=75, zeta=0.2666):
    """
    w(x) = sum_{n=0}^{n_terms-1} zeta * ( 0.05 tanh(pi x)
                                          + c^n cos(d^n pi x) e^{-|x|/2} m(x) ),
    m(x) = +1 for x < 0, -1 otherwise.  (a and b are kept for API
    compatibility; they are not used, as in the original.)
    """
    # Initialize the output tensor
    w = tf.zeros_like(x)

    for n in range(n_terms):
        # Exponentially decaying cosine part
        decay = tf.exp(-tf.abs(x) * 0.5)
        # Sum the components: tanh and decaying cosine with mirrored behavior
        w += zeta * (0.05 * tf.tanh(_PI * x) + (c ** n) * tf.cos((d ** n) * _PI * x) * decay * _mirrored_function_tf(x))

    return w


# Custom activation function: Modulated Weierstrass function
@_register
def modified_weierstrass_function_tanh(x, a=0.5, b=1.5, n_terms=100):
    """
    f(x) = [ sum_{n=0}^{n_terms-1} (-1)^n a^n cos(b^n pi x) ] e^{-0.75|x|} + tanh(x)
    computed in float64.
    """
    out_dtype = _output_dtype(x)

    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Calculate the modified Weierstrass function in float64
    w = tf.zeros_like(x, dtype=tf.float64)
    for n in range(n_terms):
        w += ((-1) ** n) * (a ** n) * tf.cos((b ** n) * _PI * x)

    # Blend with the hyperbolic tangent function
    combined_function = w * tf.exp(-0.75 * tf.abs(x)) + tf.tanh(x)

    return tf.cast(combined_function, out_dtype)


# Custom activation function: Modulated Weierstrass function with ReLU
@_register
def modified_weierstrass_function_relu(x, a=0.5, b=3, n_terms=100):
    """
    f(x) = [ sum_{n=0}^{n_terms-1} (-1)^n a^n cos(b^n pi x) ] e^{-0.75|x|} + relu(x)
    computed in float64.
    """
    out_dtype = _output_dtype(x)

    x = tf.cast(x, tf.float64)

    w = tf.zeros_like(x, dtype=tf.float64)
    for n in range(n_terms):
        w += ((-1) ** n) * (a ** n) * tf.cos((b ** n) * _PI * x)

    combined_function = w * tf.exp(-0.75 * tf.abs(x)) + tf.nn.relu(x)

    return tf.cast(combined_function, out_dtype)


@_register
def weierstrass_mandelbrot_function_xsinsquared(x, gamma=0.5, lambda_val=2, num_terms=100):
    """
    M(x) = sum_{k=1}^{num_terms-1} 2^{-k gamma} x sin^2(2 pi lambda^k x), float64.
    """
    out_dtype = _output_dtype(x)

    x = tf.cast(x, tf.float64)

    M_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, num_terms):
        term = (2**(-k * gamma)) * x * (tf.sin(2 * _PI * lambda_val**k * x) ** 2)
        M_x += term

    return tf.cast(M_x, out_dtype)


@_register
def weierstrass_mandelbrot_function_xpsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    """
    M(x) = sum_{k=1}^{num_terms-1} 2^{-k gamma} ( x + sin(2 pi lambda^k x) ), float64.
    """
    out_dtype = _output_dtype(x)

    x = tf.cast(x, tf.float64)

    M_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, num_terms):
        term = (2**(-k * gamma)) * (x + (tf.sin(2 * _PI * lambda_val**k * x)))
        M_x += term

    return tf.cast(M_x, out_dtype)


@_register
def weierstrass_mandelbrot_function_relupsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    """
    M(x) = sum_{k=1}^{num_terms-1} 2^{-k gamma} ( relu(x) + sin(2 pi lambda^k x) ), float64.
    """
    out_dtype = _output_dtype(x)

    x = tf.cast(x, tf.float64)

    M_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, num_terms):
        relu_term = tf.nn.relu(x)
        sin_term = tf.sin(2 * _PI * lambda_val**k * x)
        term = (2**(-k * gamma)) * (relu_term + sin_term)
        M_x += term

    return tf.cast(M_x, out_dtype)


@_register
def weierstrass_mandelbrot_function_tanhpsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    """
    M(x) = sum_{k=1}^{num_terms-1} 2^{-k gamma} ( tanh(x) + sin(2 pi lambda^k x) ), float64.
    """
    out_dtype = _output_dtype(x)

    x = tf.cast(x, tf.float64)

    M_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, num_terms):
        tanh_term = tf.tanh(x)
        sin_term = tf.sin(2 * _PI * lambda_val**k * x)
        term = (2**(-k * gamma)) * (tanh_term + sin_term)
        M_x += term

    return tf.cast(M_x, out_dtype)


# Custom activation function: Classical Weierstrass function
@_register
def weierstrass_function_tf(x, gamma=0.5, lambda_val=2, num_terms=100):
    """
    W(x) = sum_{k=1}^{num_terms-1} 2^{-k gamma} sin(2 pi lambda^k x), float64.
    """
    out_dtype = _output_dtype(x)

    x = tf.cast(x, tf.float64)

    W_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, num_terms):
        term = (2**(-k * gamma)) * tf.sin(2 * _PI * lambda_val**k * x)
        W_x += term

    return tf.cast(W_x, out_dtype)


# ============================================================
# Random-depth fractal activation functions
# ============================================================

def _sample_random_depth(min_terms=10, max_terms=50):
    """
    Sample one random integer depth for one activation call.

    Returns
    -------
    tf.Tensor
        Scalar int32 tensor in [min_terms, max_terms].
    """
    return tf.random.uniform(
        shape=[],
        minval=int(min_terms),
        maxval=int(max_terms) + 1,
        dtype=tf.int32,
    )


@_register
def random_depth_modified_weierstrass_function_tanh(
        x,
        a=0.5,
        b=1.5,
        min_terms=10,
        max_terms=50
    ):
    """
    Random-depth version of modified_weierstrass_function_tanh: the number of
    active series terms is sampled on each function call.
    """
    out_dtype = _output_dtype(x)
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float64)
    w = tf.zeros_like(x, dtype=tf.float64)

    for n in range(int(max_terms)):
        active = tf.cast(n < depth, tf.float64)
        w += active * ((-1) ** n) * (a ** n) * tf.cos((b ** n) * _PI * x)

    combined_function = w * tf.exp(-0.75 * tf.abs(x)) + tf.tanh(x)

    return tf.cast(combined_function, out_dtype)


@_register
def random_depth_decaying_cosine_function_tf(
        x,
        a=0.5,
        b=3,
        c=0.5,
        d=2,
        min_terms=10,
        max_terms=50,
        zeta=0.2666
    ):
    """
    Random-depth version of decaying_cosine_function_tf (computed in float32,
    as in the original).
    """
    out_dtype = _output_dtype(x)
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float32)
    w = tf.zeros_like(x, dtype=tf.float32)

    for n in range(int(max_terms)):
        active = tf.cast(n < depth, tf.float32)

        decay = tf.exp(-tf.abs(x) * 0.5)

        w += active * zeta * (
            0.05 * tf.tanh(_PI * x)
            + (c ** n)
            * tf.cos((d ** n) * _PI * x)
            * decay
            * _mirrored_function_tf(x)
        )

    return tf.cast(w, out_dtype)


@_register
def random_depth_modulated_blancmange_curve(
        x,
        min_terms=10,
        max_terms=50,
        a=0.75
    ):
    """
    Random-depth version of modulated_blancmange_curve (with 1e-8 inside the
    square root, as in the original).
    """
    out_dtype = _output_dtype(x)
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float32)
    y = tf.zeros_like(x, dtype=tf.float32)

    for n in range(int(max_terms)):
        active = tf.cast(n < depth, tf.float32)

        factor = 2 ** n
        modulation = tf.tanh(a * factor * x)
        ax = a * tf.sqrt(tf.abs(x) + 1e-8)

        y += active * modulation * tf.abs(x * factor % 2 - 1 * ax) / factor

    return tf.cast(y / 2, out_dtype)


@_register
def random_depth_weierstrass_mandelbrot_function_xpsin(
        x,
        gamma=0.5,
        lambda_val=2,
        min_terms=10,
        max_terms=50
    ):
    """
    Random-depth version of weierstrass_mandelbrot_function_xpsin.
    """
    out_dtype = _output_dtype(x)
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float64)
    M_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, int(max_terms)):
        active = tf.cast(k < depth, tf.float64)

        term = (2 ** (-k * gamma)) * (
            x + tf.sin(2 * _PI * lambda_val ** k * x)
        )

        M_x += active * term

    return tf.cast(M_x, out_dtype)


@_register
def random_depth_weierstrass_mandelbrot_function_tanhpsin(
        x,
        gamma=0.5,
        lambda_val=2,
        min_terms=10,
        max_terms=50
    ):
    """
    Random-depth version of weierstrass_mandelbrot_function_tanhpsin.
    """
    out_dtype = _output_dtype(x)
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float64)
    M_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, int(max_terms)):
        active = tf.cast(k < depth, tf.float64)

        tanh_term = tf.tanh(x)
        sin_term = tf.sin(2 * _PI * lambda_val ** k * x)

        term = (2 ** (-k * gamma)) * (tanh_term + sin_term)

        M_x += active * term

    return tf.cast(M_x, out_dtype)


@_register
def random_depth_weierstrass_function_tf(
        x,
        gamma=0.5,
        lambda_val=2,
        min_terms=10,
        max_terms=50
    ):
    """
    Random-depth version of the plain Weierstrass activation (mainly useful
    as a control activation).
    """
    out_dtype = _output_dtype(x)
    depth = _sample_random_depth(min_terms, max_terms)

    x = tf.cast(x, tf.float64)
    W_x = tf.zeros_like(x, dtype=tf.float64)

    for k in range(1, int(max_terms)):
        active = tf.cast(k < depth, tf.float64)

        term = (2 ** (-k * gamma)) * tf.sin(
            2 * _PI * lambda_val ** k * x
        )

        W_x += active * term

    return tf.cast(W_x, out_dtype)


# ============================================================
# Cantor-based activation functions
# ============================================================

@_register
def cantor_function_tf(x, depth=10):
    """
    Cantor function (Devil's staircase) of sigmoid(x), depth-d approximation,
    following Chipera, Du & Tsapara (2025), arXiv:2512.14675.
    Maps R -> [0, 1]:  f_C(x) = c_depth(sigmoid(x)),

        c_0(y) = y
        c_d(y) = 0.5 * c_{d-1}(3y)            if y in [0, 1/3]
               = 0.5                          if y in (1/3, 2/3)
               = 0.5 + 0.5 * c_{d-1}(3y - 2)  if y in [2/3, 1]

    Iterative implementation of fixed depth. The result is piecewise
    constant except for the base-case term, so its derivative is tiny or
    zero almost everywhere.
    """
    y = tf.math.sigmoid(x)          # map R -> (0, 1)

    accumulated = tf.zeros_like(y)
    scale       = tf.ones_like(y)
    current_y   = y
    active      = tf.ones_like(y)

    third      = tf.cast(1.0 / 3.0, y.dtype)
    two_thirds = tf.cast(2.0 / 3.0, y.dtype)

    for _ in range(depth):
        in_left   = active * tf.cast(current_y <= third,                                y.dtype)
        in_middle = active * tf.cast((current_y > third) & (current_y < two_thirds),   y.dtype)
        in_right  = active * tf.cast(current_y >= two_thirds,                          y.dtype)

        # Middle third: function is 0.5 at this level — freeze these points.
        accumulated = accumulated + in_middle * 0.5 * scale
        active      = active - in_middle

        # Right third: contribute 0.5 * scale, then recurse into (3y - 2).
        accumulated = accumulated + in_right * 0.5 * scale

        # Rescale for the next level.
        current_y = in_left * (current_y * 3.0) + in_right * (current_y * 3.0 - 2.0)
        scale     = scale * 0.5

    # Base case c_0(y) = y: remaining active points contribute scale * current_y.
    accumulated = accumulated + active * scale * current_y

    return accumulated


@_register
def cantor_set_activation_tf(x, depth=10):
    """
    Cantor-set indicator activation (cf. Chipera, Du & Tsapara (2025),
    arXiv:2512.14675). Indicator of the depth-d Cantor set applied to
    sigmoid(x): returns 1 if no ternary digit (positions 1..depth) of
    sigmoid(x) equals 1, else 0.
    Binary output {0, 1}; zero derivative everywhere.
    """
    y = tf.math.sigmoid(x)          # map R -> (0, 1)

    in_cantor = tf.ones_like(y)
    current_y = y

    third      = tf.cast(1.0 / 3.0, y.dtype)
    two_thirds = tf.cast(2.0 / 3.0, y.dtype)

    for _ in range(depth):
        is_middle = tf.cast((current_y >= third) & (current_y < two_thirds), y.dtype)

        # Points in the middle third are excluded from the Cantor set.
        in_cantor = in_cantor * (1.0 - is_middle)

        current_y = tf.where(
            current_y < third,
            current_y * 3.0,
            tf.where(
                current_y >= two_thirds,
                current_y * 3.0 - 2.0,
                current_y,              # middle: keep as-is (already excluded)
            ),
        )

    return in_cantor


@_register
def random_depth_cantor_function_tf(x, min_depth=5, max_depth=15):
    """
    Cantor function of sigmoid(x) with a recursion depth sampled uniformly
    from {min_depth, ..., max_depth} on each call (one scalar depth per call).
    The Cantor-function activation itself follows Chipera, Du & Tsapara
    (2025), arXiv:2512.14675; the random depth is an extension in this code base.
    """
    depth = _sample_random_depth(min_terms=min_depth, max_terms=max_depth)

    y = tf.math.sigmoid(x)

    accumulated = tf.zeros_like(y)
    scale       = tf.ones_like(y)
    current_y   = y
    active      = tf.ones_like(y)

    third      = tf.cast(1.0 / 3.0, y.dtype)
    two_thirds = tf.cast(2.0 / 3.0, y.dtype)

    for step in range(max_depth):
        # Gate: 1.0 if this step is within the sampled depth, 0.0 otherwise.
        step_gate = tf.cast(step < depth, y.dtype)

        in_left   = active * tf.cast(current_y <= third,                                y.dtype)
        in_middle = active * tf.cast((current_y > third) & (current_y < two_thirds),   y.dtype)
        in_right  = active * tf.cast(current_y >= two_thirds,                          y.dtype)

        accumulated = accumulated + step_gate * in_middle * 0.5 * scale
        active      = active      - step_gate * in_middle
        accumulated = accumulated + step_gate * in_right  * 0.5 * scale

        new_y     = in_left * (current_y * 3.0) + in_right * (current_y * 3.0 - 2.0)
        current_y = step_gate * new_y + (1.0 - step_gate) * current_y

        scale     = scale * (step_gate * 0.5 + (1.0 - step_gate) * 1.0)

    accumulated = accumulated + active * scale * current_y

    return accumulated


# ============================================================
# Registries and a Keras layer wrapper
# ============================================================

# The four fractal activations used in the article.
PAPER_FRACTAL_ACTIVATIONS: Dict[str, Callable] = {
    "modulated_blancmange_curve": modulated_blancmange_curve,
    "decaying_cosine_function_tf": decaying_cosine_function_tf,
    "modified_weierstrass_function_tanh": modified_weierstrass_function_tanh,
    "weierstrass_mandelbrot_function_xpsin": weierstrass_mandelbrot_function_xpsin,
}

# Every activation in this module.
ALL_FRACTAL_ACTIVATIONS: Dict[str, Callable] = {
    "modulated_blancmange_curve": modulated_blancmange_curve,
    "decaying_cosine_function_tf": decaying_cosine_function_tf,
    "modified_weierstrass_function_tanh": modified_weierstrass_function_tanh,
    "modified_weierstrass_function_relu": modified_weierstrass_function_relu,
    "weierstrass_mandelbrot_function_xsinsquared": weierstrass_mandelbrot_function_xsinsquared,
    "weierstrass_mandelbrot_function_xpsin": weierstrass_mandelbrot_function_xpsin,
    "weierstrass_mandelbrot_function_relupsin": weierstrass_mandelbrot_function_relupsin,
    "weierstrass_mandelbrot_function_tanhpsin": weierstrass_mandelbrot_function_tanhpsin,
    "weierstrass_function_tf": weierstrass_function_tf,
    "random_depth_modified_weierstrass_function_tanh": random_depth_modified_weierstrass_function_tanh,
    "random_depth_decaying_cosine_function_tf": random_depth_decaying_cosine_function_tf,
    "random_depth_modulated_blancmange_curve": random_depth_modulated_blancmange_curve,
    "random_depth_weierstrass_mandelbrot_function_xpsin": random_depth_weierstrass_mandelbrot_function_xpsin,
    "random_depth_weierstrass_mandelbrot_function_tanhpsin": random_depth_weierstrass_mandelbrot_function_tanhpsin,
    "random_depth_weierstrass_function_tf": random_depth_weierstrass_function_tf,
    "cantor_function_tf": cantor_function_tf,
    "cantor_set_activation_tf": cantor_set_activation_tf,
    "random_depth_cantor_function_tf": random_depth_cantor_function_tf,
}


def get_fractal_activation(name: str) -> Callable:
    """Look up an activation function by name."""
    try:
        return ALL_FRACTAL_ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown fractal activation {name!r}. "
            f"Available: {sorted(ALL_FRACTAL_ACTIVATIONS)}"
        ) from None


@keras.saving.register_keras_serializable(package="FractalActivations")
class FractalActivation(keras.layers.Layer):
    """
    Keras layer that applies one of the fractal activations element-wise,
    optionally with non-default parameters.

        x = keras.layers.Dense(32)(x)
        x = FractalActivation("modified_weierstrass_function_tanh", n_terms=50)(x)

    is the same computation as Dense(32, activation=...) with that function,
    and the parameters are stored in the model config, so the model can be
    saved and loaded.
    """

    def __init__(self, function_name: str, function_kwargs: dict = None, **kwargs):
        # Allow FractalActivation("name", n_terms=50) as well as
        # FractalActivation("name", function_kwargs={"n_terms": 50}).
        layer_kwarg_names = {
            "name", "dtype", "trainable", "autocast", "input_shape", "input_dim",
            "batch_input_shape", "batch_size", "weights", "activity_regularizer",
        }
        extra = {k: kwargs.pop(k) for k in list(kwargs) if k not in layer_kwarg_names}
        super().__init__(**kwargs)
        self.function_name = str(function_name)
        self.function_kwargs = dict(function_kwargs or {})
        self.function_kwargs.update(extra)
        self._fn = get_fractal_activation(self.function_name)
        self.supports_masking = True

    def call(self, inputs):
        return self._fn(inputs, **self.function_kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "function_name": self.function_name,
                "function_kwargs": dict(self.function_kwargs),
            }
        )
        return config


__all__ = sorted(ALL_FRACTAL_ACTIVATIONS) + [
    "PAPER_FRACTAL_ACTIVATIONS",
    "ALL_FRACTAL_ACTIVATIONS",
    "get_fractal_activation",
    "FractalActivation",
]
