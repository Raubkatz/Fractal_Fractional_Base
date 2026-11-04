"""
fractal_activation_functions_torch.py

Fractal activation functions for PyTorch (tested with PyTorch 2.15).

This file is the PyTorch counterpart of `fractal_activation_functions_keras3.py`.
Every function keeps its name, its signature, its default parameters and its
arithmetic (same operations in the same order, same internal precision):

- the Weierstrass-type functions compute in float64 and return the dtype of
  their input,
- modulated_blancmange_curve, decaying_cosine_function_tf and the Cantor
  functions compute in the dtype of their input,
- the random-depth blancmange / decaying-cosine variants compute in float32,
- Python's `%` on tensors is torch.remainder (floor modulo, same as tf.math.floormod),
- the random depth is one scalar integer per call, drawn with torch.randint on
  the input's device; terms beyond the sampled depth are multiplied by 0
  (the same masked loop as in the Keras version, so it also works under
  torch.compile and on the GPU without a host synchronisation).

The function names keep their historical "_tf" suffixes so that the same names
work in both frameworks.

Usage
-----
    import torch
    import fractal_activation_functions_torch as faf

    # 1) as a function inside forward()
    h = faf.modified_weierstrass_function_tanh(self.fc1(x))

    # 2) as a module (default or non-default parameters), e.g. in nn.Sequential
    torch.nn.Sequential(
        torch.nn.Linear(4, 32),
        faf.FractalActivation("modified_weierstrass_function_tanh", n_terms=50),
    )

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

Numerical notes (same behaviour as the original functions)
----------------------------------------------------------
- modulated_blancmange_curve contains sqrt(|x|), whose derivative is
  unbounded at x = 0; the random-depth variant adds 1e-8 inside the sqrt.
  At exactly x = 0 the Keras gradient is 0 (TensorFlow's sqrt gradient is 0
  when the incoming gradient is 0); this file uses the same rule
  (_SqrtWithTFGradient), because plain torch.sqrt would give NaN there.
- cantor_function_tf and cantor_set_activation_tf are piecewise constant at
  every finite depth (zero derivative almost everywhere), so gradients do
  not pass through them: hidden layers using them do not learn.
- Python integer powers such as 2**n and lambda_val**k are evaluated in
  Python and then multiplied into the tensor, exactly as in the Keras file.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import torch

_PI = math.pi  # identical to np.pi / tnp.pi


def _as_float_tensor(x) -> torch.Tensor:
    """Tensor input unchanged; anything else becomes a tensor; non-float -> float32."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if not x.is_floating_point():
        x = x.to(torch.float32)
    return x


class _SqrtWithTFGradient(torch.autograd.Function):
    """
    sqrt(x) with TensorFlow's gradient rule (SqrtGrad): the gradient is 0
    wherever the incoming gradient is 0, otherwise 0.5 * grad / sqrt(x).

    PyTorch's own rule gives 0 / 0 = NaN at x = 0 even when the incoming
    gradient is 0. In modulated_blancmange_curve the sqrt(|x|) term is
    multiplied by tanh(0) = 0 at x = 0, so TensorFlow returns a gradient of 0
    there and plain torch.sqrt would return NaN (which would poison the whole
    network). With this rule the PyTorch gradient equals the Keras gradient,
    also at x = 0.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        return torch.sqrt(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return torch.where(grad == 0, torch.zeros_like(grad), 0.5 * grad / y)


def _sqrt(x):
    return _SqrtWithTFGradient.apply(x)


# ============================================================
# Fixed-depth fractal activation functions
# ============================================================


# Custom activation function: Revised Modulated Blancmange Curve
def modulated_blancmange_curve(x, n_terms=30, a=0.75):
    """
    y(x) = 1/2 * sum_{n=0}^{n_terms-1} tanh(a 2^n x) * | (2^n x mod 2) - a sqrt|x| | / 2^n
    """
    x = _as_float_tensor(x)
    y = torch.zeros_like(x)
    for n in range(n_terms):
        factor = 2 ** n
        modulation = torch.tanh(a * factor * x)  # Modulating factor using tanh
        ax = a * _sqrt(torch.abs(x))          # sqrt with TensorFlow's gradient at 0
        y = y + modulation * torch.abs(torch.remainder(x * factor, 2) - 1 * ax) / factor
    return y / 2


def _mirrored_function_tf(x):
    """+1 for x < 0, -1 otherwise (used by the decaying cosine functions)."""
    return torch.where(x < 0, torch.ones_like(x), -torch.ones_like(x))


# Blended function with tanh and cosine components with exponential decay
def decaying_cosine_function_tf(x, a=0.5, b=3, c=0.5, d=2, n_terms=75, zeta=0.2666):
    """
    w(x) = sum_{n=0}^{n_terms-1} zeta * ( 0.05 tanh(pi x)
                                          + c^n cos(d^n pi x) e^{-|x|/2} m(x) ),
    m(x) = +1 for x < 0, -1 otherwise.  (a and b are kept for API
    compatibility; they are not used, as in the original.)
    """
    x = _as_float_tensor(x)
    w = torch.zeros_like(x)

    for n in range(n_terms):
        # Exponentially decaying cosine part
        decay = torch.exp(-torch.abs(x) * 0.5)
        # Sum the components: tanh and decaying cosine with mirrored behavior
        w = w + zeta * (0.05 * torch.tanh(_PI * x)
                        + (c ** n) * torch.cos((d ** n) * _PI * x) * decay * _mirrored_function_tf(x))

    return w


# Custom activation function: Modulated Weierstrass function
def modified_weierstrass_function_tanh(x, a=0.5, b=1.5, n_terms=100):
    """
    f(x) = [ sum_{n=0}^{n_terms-1} (-1)^n a^n cos(b^n pi x) ] e^{-0.75|x|} + tanh(x)
    computed in float64.
    """
    x = _as_float_tensor(x)
    out_dtype = x.dtype

    # Ensure x is float64 for all calculations
    x = x.to(torch.float64)

    # Calculate the modified Weierstrass function in float64
    w = torch.zeros_like(x)
    for n in range(n_terms):
        w = w + ((-1) ** n) * (a ** n) * torch.cos((b ** n) * _PI * x)

    # Blend with the hyperbolic tangent function
    combined_function = w * torch.exp(-0.75 * torch.abs(x)) + torch.tanh(x)

    return combined_function.to(out_dtype)


# Custom activation function: Modulated Weierstrass function with ReLU
def modified_weierstrass_function_relu(x, a=0.5, b=3, n_terms=100):
    """
    f(x) = [ sum_{n=0}^{n_terms-1} (-1)^n a^n cos(b^n pi x) ] e^{-0.75|x|} + relu(x)
    computed in float64.
    """
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    x = x.to(torch.float64)

    w = torch.zeros_like(x)
    for n in range(n_terms):
        w = w + ((-1) ** n) * (a ** n) * torch.cos((b ** n) * _PI * x)

    combined_function = w * torch.exp(-0.75 * torch.abs(x)) + torch.relu(x)

    return combined_function.to(out_dtype)


def weierstrass_mandelbrot_function_xsinsquared(x, gamma=0.5, lambda_val=2, num_terms=100):
    """
    M(x) = sum_{k=1}^{num_terms-1} 2^{-k gamma} x sin^2(2 pi lambda^k x), float64.
    """
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    x = x.to(torch.float64)

    M_x = torch.zeros_like(x)
    for k in range(1, num_terms):
        term = (2 ** (-k * gamma)) * x * (torch.sin(2 * _PI * lambda_val ** k * x) ** 2)
        M_x = M_x + term

    return M_x.to(out_dtype)


def weierstrass_mandelbrot_function_xpsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    """
    M(x) = sum_{k=1}^{num_terms-1} 2^{-k gamma} ( x + sin(2 pi lambda^k x) ), float64.
    """
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    x = x.to(torch.float64)

    M_x = torch.zeros_like(x)
    for k in range(1, num_terms):
        term = (2 ** (-k * gamma)) * (x + (torch.sin(2 * _PI * lambda_val ** k * x)))
        M_x = M_x + term

    return M_x.to(out_dtype)


def weierstrass_mandelbrot_function_relupsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    """
    M(x) = sum_{k=1}^{num_terms-1} 2^{-k gamma} ( relu(x) + sin(2 pi lambda^k x) ), float64.
    """
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    x = x.to(torch.float64)

    M_x = torch.zeros_like(x)
    for k in range(1, num_terms):
        relu_term = torch.relu(x)
        sin_term = torch.sin(2 * _PI * lambda_val ** k * x)
        term = (2 ** (-k * gamma)) * (relu_term + sin_term)
        M_x = M_x + term

    return M_x.to(out_dtype)


def weierstrass_mandelbrot_function_tanhpsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    """
    M(x) = sum_{k=1}^{num_terms-1} 2^{-k gamma} ( tanh(x) + sin(2 pi lambda^k x) ), float64.
    """
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    x = x.to(torch.float64)

    M_x = torch.zeros_like(x)
    for k in range(1, num_terms):
        tanh_term = torch.tanh(x)
        sin_term = torch.sin(2 * _PI * lambda_val ** k * x)
        term = (2 ** (-k * gamma)) * (tanh_term + sin_term)
        M_x = M_x + term

    return M_x.to(out_dtype)


# Custom activation function: Classical Weierstrass function
def weierstrass_function_tf(x, gamma=0.5, lambda_val=2, num_terms=100):
    """
    W(x) = sum_{k=1}^{num_terms-1} 2^{-k gamma} sin(2 pi lambda^k x), float64.
    """
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    x = x.to(torch.float64)

    W_x = torch.zeros_like(x)
    for k in range(1, num_terms):
        term = (2 ** (-k * gamma)) * torch.sin(2 * _PI * lambda_val ** k * x)
        W_x = W_x + term

    return W_x.to(out_dtype)


# ============================================================
# Random-depth fractal activation functions
# ============================================================

def _sample_random_depth(min_terms=10, max_terms=50, device=None):
    """
    Sample one random integer depth for one activation call.

    Returns
    -------
    torch.Tensor
        Scalar int64 tensor in [min_terms, max_terms] on `device`.
    """
    return torch.randint(int(min_terms), int(max_terms) + 1, (), device=device)


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
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    depth = _sample_random_depth(min_terms, max_terms, device=x.device)

    x = x.to(torch.float64)
    w = torch.zeros_like(x)

    for n in range(int(max_terms)):
        active = (depth > n).to(torch.float64)          # 1.0 while n < depth
        w = w + active * ((-1) ** n) * (a ** n) * torch.cos((b ** n) * _PI * x)

    combined_function = w * torch.exp(-0.75 * torch.abs(x)) + torch.tanh(x)

    return combined_function.to(out_dtype)


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
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    depth = _sample_random_depth(min_terms, max_terms, device=x.device)

    x = x.to(torch.float32)
    w = torch.zeros_like(x)

    for n in range(int(max_terms)):
        active = (depth > n).to(torch.float32)

        decay = torch.exp(-torch.abs(x) * 0.5)

        w = w + active * zeta * (
            0.05 * torch.tanh(_PI * x)
            + (c ** n)
            * torch.cos((d ** n) * _PI * x)
            * decay
            * _mirrored_function_tf(x)
        )

    return w.to(out_dtype)


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
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    depth = _sample_random_depth(min_terms, max_terms, device=x.device)

    x = x.to(torch.float32)
    y = torch.zeros_like(x)

    for n in range(int(max_terms)):
        active = (depth > n).to(torch.float32)

        factor = 2 ** n
        modulation = torch.tanh(a * factor * x)
        ax = a * torch.sqrt(torch.abs(x) + 1e-8)

        y = y + active * modulation * torch.abs(torch.remainder(x * factor, 2) - 1 * ax) / factor

    return (y / 2).to(out_dtype)


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
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    depth = _sample_random_depth(min_terms, max_terms, device=x.device)

    x = x.to(torch.float64)
    M_x = torch.zeros_like(x)

    for k in range(1, int(max_terms)):
        active = (depth > k).to(torch.float64)

        term = (2 ** (-k * gamma)) * (
            x + torch.sin(2 * _PI * lambda_val ** k * x)
        )

        M_x = M_x + active * term

    return M_x.to(out_dtype)


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
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    depth = _sample_random_depth(min_terms, max_terms, device=x.device)

    x = x.to(torch.float64)
    M_x = torch.zeros_like(x)

    for k in range(1, int(max_terms)):
        active = (depth > k).to(torch.float64)

        tanh_term = torch.tanh(x)
        sin_term = torch.sin(2 * _PI * lambda_val ** k * x)

        term = (2 ** (-k * gamma)) * (tanh_term + sin_term)

        M_x = M_x + active * term

    return M_x.to(out_dtype)


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
    x = _as_float_tensor(x)
    out_dtype = x.dtype
    depth = _sample_random_depth(min_terms, max_terms, device=x.device)

    x = x.to(torch.float64)
    W_x = torch.zeros_like(x)

    for k in range(1, int(max_terms)):
        active = (depth > k).to(torch.float64)

        term = (2 ** (-k * gamma)) * torch.sin(
            2 * _PI * lambda_val ** k * x
        )

        W_x = W_x + active * term

    return W_x.to(out_dtype)


# ============================================================
# Cantor-based activation functions
# ============================================================

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
    x = _as_float_tensor(x)
    y = torch.sigmoid(x)            # map R -> (0, 1)

    accumulated = torch.zeros_like(y)
    scale       = torch.ones_like(y)
    current_y   = y
    active      = torch.ones_like(y)

    third      = torch.tensor(1.0 / 3.0, dtype=y.dtype, device=y.device)
    two_thirds = torch.tensor(2.0 / 3.0, dtype=y.dtype, device=y.device)

    for _ in range(depth):
        in_left   = active * (current_y <= third).to(y.dtype)
        in_middle = active * ((current_y > third) & (current_y < two_thirds)).to(y.dtype)
        in_right  = active * (current_y >= two_thirds).to(y.dtype)

        # Middle third: function is 0.5 at this level - freeze these points.
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


def cantor_set_activation_tf(x, depth=10):
    """
    Cantor-set indicator activation (cf. Chipera, Du & Tsapara (2025),
    arXiv:2512.14675). Indicator of the depth-d Cantor set applied to
    sigmoid(x): returns 1 if no ternary digit (positions 1..depth) of
    sigmoid(x) equals 1, else 0.
    Binary output {0, 1}; zero derivative everywhere. As in the Keras version
    the output is built from comparisons only, so it carries no gradient path
    (layers before it get grad None, which torch optimizers skip).
    """
    x = _as_float_tensor(x)
    y = torch.sigmoid(x)            # map R -> (0, 1)

    in_cantor = torch.ones_like(y)
    current_y = y

    third      = torch.tensor(1.0 / 3.0, dtype=y.dtype, device=y.device)
    two_thirds = torch.tensor(2.0 / 3.0, dtype=y.dtype, device=y.device)

    for _ in range(depth):
        is_middle = ((current_y >= third) & (current_y < two_thirds)).to(y.dtype)

        # Points in the middle third are excluded from the Cantor set.
        in_cantor = in_cantor * (1.0 - is_middle)

        current_y = torch.where(
            current_y < third,
            current_y * 3.0,
            torch.where(
                current_y >= two_thirds,
                current_y * 3.0 - 2.0,
                current_y,              # middle: keep as-is (already excluded)
            ),
        )

    return in_cantor


def random_depth_cantor_function_tf(x, min_depth=5, max_depth=15):
    """
    Cantor function of sigmoid(x) with a recursion depth sampled uniformly
    from {min_depth, ..., max_depth} on each call (one scalar depth per call).
    The Cantor-function activation itself follows Chipera, Du & Tsapara
    (2025), arXiv:2512.14675; the random depth is an extension in this code base.
    """
    x = _as_float_tensor(x)
    depth = _sample_random_depth(min_terms=min_depth, max_terms=max_depth, device=x.device)

    y = torch.sigmoid(x)

    accumulated = torch.zeros_like(y)
    scale       = torch.ones_like(y)
    current_y   = y
    active      = torch.ones_like(y)

    third      = torch.tensor(1.0 / 3.0, dtype=y.dtype, device=y.device)
    two_thirds = torch.tensor(2.0 / 3.0, dtype=y.dtype, device=y.device)

    for step in range(max_depth):
        # Gate: 1.0 if this step is within the sampled depth, 0.0 otherwise.
        step_gate = (depth > step).to(y.dtype)

        in_left   = active * (current_y <= third).to(y.dtype)
        in_middle = active * ((current_y > third) & (current_y < two_thirds)).to(y.dtype)
        in_right  = active * (current_y >= two_thirds).to(y.dtype)

        accumulated = accumulated + step_gate * in_middle * 0.5 * scale
        active      = active      - step_gate * in_middle
        accumulated = accumulated + step_gate * in_right  * 0.5 * scale

        new_y     = in_left * (current_y * 3.0) + in_right * (current_y * 3.0 - 2.0)
        current_y = step_gate * new_y + (1.0 - step_gate) * current_y

        scale     = scale * (step_gate * 0.5 + (1.0 - step_gate) * 1.0)

    accumulated = accumulated + active * scale * current_y

    return accumulated


# ============================================================
# Registries and an nn.Module wrapper
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


class FractalActivation(torch.nn.Module):
    """
    nn.Module that applies one of the fractal activations element-wise,
    optionally with non-default parameters.

        torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            FractalActivation("modified_weierstrass_function_tanh", n_terms=50),
        )

    is the same computation as calling the function on the Linear output.
    The module has no parameters; its settings appear in print(model).
    """

    def __init__(self, function_name: str, function_kwargs: dict = None, **kwargs):
        # Allow FractalActivation("name", n_terms=50) as well as
        # FractalActivation("name", function_kwargs={"n_terms": 50}).
        super().__init__()
        self.function_name = str(function_name)
        self.function_kwargs = dict(function_kwargs or {})
        self.function_kwargs.update(kwargs)
        self._fn = get_fractal_activation(self.function_name)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._fn(inputs, **self.function_kwargs)

    def extra_repr(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.function_kwargs.items())
        return f"{self.function_name!r}" + (f", {args}" if args else "")


__all__ = sorted(ALL_FRACTAL_ACTIVATIONS) + [
    "PAPER_FRACTAL_ACTIVATIONS",
    "ALL_FRACTAL_ACTIVATIONS",
    "get_fractal_activation",
    "FractalActivation",
]


if __name__ == "__main__":
    xs = torch.linspace(-3.0, 3.0, 13, requires_grad=True)
    for name, fn in ALL_FRACTAL_ACTIVATIONS.items():
        out = fn(xs)
        if out.requires_grad:              # cantor_set_activation_tf has no gradient path
            out.sum().backward()
            xs.grad = None
        finite = bool(torch.isfinite(out).all())
        print(f"  {name:55s} dtype={str(out.dtype):14s} finite={finite}  "
              f"f(1)={float(fn(torch.tensor([1.0]))[0]):+.6f}")
    print(FractalActivation("modified_weierstrass_function_tanh", n_terms=50))
    print("All activations evaluated.")
