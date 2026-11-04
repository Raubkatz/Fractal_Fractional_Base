#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_tests_pytorch_vs_tensorflow.py
=================================

Cross-framework check: every PyTorch optimizer and activation against its
TensorFlow / Keras 3 twin, with identical inputs. Written top to bottom,
step by step.

STEP  0  Settings
STEP  1  Imports (PyTorch BEFORE TensorFlow, see the note there)
STEP  2  The toy problem: a 4-8-3 tanh network, loss and gradients in NumPy
STEP  3  The 167 optimizer configurations (the same list as in 06_tests_tensorflow_vs_legacy.py)
STEP  4  A recorded gradient sequence for the open-loop experiment
STEP  5  One training run in Keras 3 and one in PyTorch (two functions)
STEP  6  Run every configuration in both frameworks
STEP  7  Compare the optimizers
STEP  8  Compare the fractal activations (values and gradients)
STEP  9  Summary

Two optimizer experiments (60 steps each, same initial weights):
  open loop    both optimizers receive the same recorded sequence of gradients
               and losses, so only the optimizer arithmetic is compared;
  closed loop  each optimizer trains the network on its own weights (normal
               training; loss and gradients from one shared NumPy function).

Pass criteria
  float64: max |w_torch - w_keras| / max |w_keras| (and the same for the
           final lambda_t / alpha_t and the loss curve) <= 1e-12 (open loop)
           and <= 1e-10 (closed loop). TensorFlow and PyTorch use different
           math libraries (pow, exp, sqrt, lgamma differ by 1-4 units in the
           last place), so bit-identical results are not possible.
           Note: tf.cast(0.9, tf.float64) rounds through float32, so in
           float64 the Keras code would run with float32-rounded
           hyperparameters. During the Keras float64 runs, this script
           converts Python numbers straight to float64. The optimizer code
           is not touched, and nothing changes in float32.
  float32: float32 round-off is amplified by some updates (most of all by
           AOFGD's (|theta| + eps)^(1 - alpha) for weights near zero), so the
           PyTorch-vs-Keras difference is compared with the float32 error of
           the Keras version itself:
               |torch32 - keras32| <= max(10 * |keras32 - keras64|, 1e-5).
  For the Bernoulli p = 0.5 runs of FCSGD_GL / FCAdam_GL, tf.random.uniform
  and torch.rand are replaced by the same seeded NumPy stream, and both
  frameworks must draw one fresh mask per variable and step.
  Keras keeps its learning rate in a float32 variable, so the PyTorch
  optimizers get the float32-rounded learning rate.

Activations: 20,001 points in [-5, 5] and a random 64 x 32 batch, float32
and float64 inputs; the random-depth functions with the depth fixed to its
minimum, middle and maximum value. Output dtype, gradient path and NaN/inf
positions must agree exactly; values and gradients (relative to the largest
Keras value) <= 1e-12 for float64 inputs, <= 1e-6 for float32 inputs of the
functions that compute in float64 (Weierstrass family), <= 2e-5 for the
functions that compute in float32. One documented exception:
random_depth_cantor_function_tf in float32 at its maximum depth (15 ternary
levels multiply the last-bit difference of the sigmoid by 3^15): values
<= 1e-4 and at most 0.5 % of the gradients may differ by more than 1e-3.

Run it with
    python 05_tests_pytorch_vs_tensorflow.py

(no command-line arguments: every setting is in STEP 0). Needs the Keras 3
and the PyTorch files in the same folder, and both TensorFlow and PyTorch.
Takes about 10 minutes on a 2-core CPU.
"""

# =============================================================================
# STEP 0  SETTINGS
# =============================================================================

RUN_OPTIMIZER_CHECK = True
RUN_ACTIVATION_CHECK = True
DTYPES = ("float64", "float32")    # float32 also runs Keras in float64 as its reference
N_STEPS = 60                       # training steps per configuration
TOLERANCE_FLOAT64 = {"open": 1e-12, "closed": 1e-10}
FLOAT32_FACTOR, FLOAT32_FLOOR = 10.0, 1e-5
PRINT_EVERY_CONFIGURATION = True   # one line per configuration in STEP 7


# =============================================================================
# STEP 1  IMPORTS
# =============================================================================
# PyTorch (and torch._dynamo, which loads Triton) MUST be imported before
# TensorFlow. With the default Linux PyTorch wheel, TensorFlow first and then
# a PyTorch optimizer step aborts the process with "LLVM ERROR: inconsistency
# in registered CommandLine options" (both bundle LLVM). In this order both
# frameworks work side by side in one process.

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import torch
import torch._dynamo  # noqa: F401  (must come before tensorflow, see above)

import numpy as np
import tensorflow as tf
import keras

import class_b_tf_fractional_optimizers_OHA_keras3
import class_c_tf_fractional_optimizers_SR_keras3
import class_minimal_fractional_adaptive_optimizers_keras3
import class_d_related_work_optimizers_keras3
import fractal_activation_functions_keras3 as faf_keras

import class_b_fractional_optimizers_OHA_torch
import class_c_fractional_optimizers_SR_torch
import class_minimal_fractional_adaptive_optimizers_torch
import class_d_related_work_optimizers_torch
import fractal_activation_functions_torch as faf_torch

KERAS_MODULES = {
    "B": class_b_tf_fractional_optimizers_OHA_keras3,
    "C": class_c_tf_fractional_optimizers_SR_keras3,
    "M": class_minimal_fractional_adaptive_optimizers_keras3,
    "D": class_d_related_work_optimizers_keras3,
}
TORCH_MODULES = {
    "B": class_b_fractional_optimizers_OHA_torch,
    "C": class_c_fractional_optimizers_SR_torch,
    "M": class_minimal_fractional_adaptive_optimizers_torch,
    "D": class_d_related_work_optimizers_torch,
}
start_time = time.time()
print(f"TensorFlow {tf.__version__}, Keras {keras.__version__}, PyTorch {torch.__version__}")


# =============================================================================
# STEP 2  THE TOY PROBLEM
# =============================================================================
# 64 samples, a 4-8-3 tanh network, softmax cross-entropy, mini-batches of 16
# cycled in a fixed order. Loss and gradients come from ONE NumPy function,
# used by both frameworks.

rng = np.random.default_rng(0)
X = rng.normal(size=(64, 4)).astype(np.float32)
Y = rng.integers(0, 3, size=(64,)).astype(np.int32)
INITIAL_WEIGHTS = [
    (rng.normal(size=(4, 8)) * 0.7).astype(np.float32),
    (rng.normal(size=(8,)) * 0.1).astype(np.float32),
    (rng.normal(size=(8, 3)) * 0.7).astype(np.float32),
    (rng.normal(size=(3,)) * 0.1).astype(np.float32),
]
N_VARIABLES = len(INITIAL_WEIGHTS)


def loss_and_gradients(weights, x_batch, y_batch, np_dtype):
    """Softmax cross-entropy of the 4-8-3 tanh network and its gradients (NumPy)."""
    w0, b0, w2, b2 = weights
    h = np.tanh(x_batch @ w0 + b0)
    logits = h @ w2 + b2
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    n = len(y_batch)
    loss = float(-np.mean(np.log(p[np.arange(n), y_batch])))
    d = p.copy()
    d[np.arange(n), y_batch] -= 1.0
    d /= n
    dh = (d @ w2.T) * (1.0 - h * h)
    gradients = [x_batch.T @ dh, dh.sum(axis=0), h.T @ d, d.sum(axis=0)]
    return loss, [g.astype(np_dtype) for g in gradients]


# =============================================================================
# STEP 3  THE 167 OPTIMIZER CONFIGURATIONS
# =============================================================================
# Every class; nu < 1, nu = 1 (classical limit) and nu > 1; several history
# lengths; momentum, Nesterov, centered and AMSGrad variants; the memory
# switches; warm-up; loss feedback; Bernoulli masks with p = 1 and p = 0.5.
# ("modes" is only used by 06_tests_tensorflow_vs_legacy.py.)

CONFIGS = []


def add(module, cls, tag="", modes=("graph", "eager"), loss_feedback=False, **kwargs):
    cid = f"{cls}{('[' + tag + ']') if tag else ''}"
    CONFIGS.append(dict(id=cid, module=module, cls=cls, kwargs=kwargs,
                        modes=modes, loss_feedback=loss_feedback))


# ---------------------------------------------------------------------------
# class_b: Herrera-style
# ---------------------------------------------------------------------------
for nu in (0.75, 1.25):
    add("B", "FSGD", f"nu={nu}", lr=1e-2, vderiv=nu)
    add("B", "FSGD", f"nu={nu},mom", lr=1e-2, momentum=0.9, vderiv=nu)
    add("B", "FSGD", f"nu={nu},nesterov", lr=1e-2, momentum=0.9, nesterov=True, vderiv=nu)
    add("B", "FAdam", f"nu={nu}", lr=1e-2, vderiv=nu)
    add("B", "FAdam", f"nu={nu},amsgrad", lr=1e-2, amsgrad=True, vderiv=nu)
    add("B", "FAdagrad", f"nu={nu}", lr=1e-1, vderiv=nu)
    add("B", "FAdadelta", f"nu={nu}", lr=1.0, vderiv=nu)
    add("B", "FRMSprop", f"nu={nu}", lr=1e-2, vderiv=nu)
    add("B", "FRMSprop", f"nu={nu},mom,centered", lr=1e-2, momentum=0.5, centered=True, vderiv=nu)
    add("B", "FAdamW", f"nu={nu}", lr=1e-2, weight_decay=0.01, vderiv=nu)
    add("B", "FAdamW", f"nu={nu},amsgrad", lr=1e-2, weight_decay=0.01, amsgrad=True, vderiv=nu)

# ---------------------------------------------------------------------------
# class_c: explicit memory (GL) family
# ---------------------------------------------------------------------------
C_CLASSES = ["MemoryFSGD", "MemoryFRMSprop", "MemoryFAdam", "MemoryFAdadelta",
             "MemoryFAdagrad", "MemoryFAdamW", "MemoryFNadam", "MemoryFAdamax"]
C_LR = {"MemoryFSGD": 5e-2, "MemoryFRMSprop": 1e-2, "MemoryFAdam": 1e-2,
        "MemoryFAdadelta": 1.0, "MemoryFAdagrad": 1e-1, "MemoryFAdamW": 1e-2,
        "MemoryFNadam": 1e-2, "MemoryFAdamax": 1e-2}
for cls in C_CLASSES:
    for nu in (0.75, 1.5):
        for k in (4, 6):
            add("C", cls, f"nu={nu},K={k}", lr=C_LR[cls], vderiv=nu, history_size=k)
    add("C", cls, "nu=1(classical)", lr=C_LR[cls], vderiv=1.0, history_size=6)
    add("C", cls, "nu=0.9,K=6,no-norm-match,no-descent-check,frac-2nd-moment",
        lr=C_LR[cls], vderiv=0.9, history_size=6, norm_match=False,
        descent_check=False, use_raw_gradient_for_second_moment=False)
    add("C", cls, "nu=1.25,K=6,normalized", lr=C_LR[cls], vderiv=1.25, history_size=6,
        normalize_coefficients=True)
add("C", "MemoryFSGD", "nu=0.75,K=6,mom,nesterov", lr=5e-2, momentum=0.9, nesterov=True, vderiv=0.75, history_size=6)
add("C", "MemoryFSGD", "nu=0.75,K=6,mom", lr=5e-2, momentum=0.9, vderiv=0.75, history_size=6)
add("C", "MemoryFRMSprop", "nu=0.75,K=6,mom,centered", lr=1e-2, momentum=0.5, centered=True, vderiv=0.75, history_size=6)
add("C", "MemoryFAdam", "nu=0.75,K=6,amsgrad", lr=1e-2, amsgrad=True, vderiv=0.75, history_size=6)
add("C", "MemoryFAdamW", "nu=1.25,K=6,wd=0.02", lr=1e-2, weight_decay=0.02, vderiv=1.25, history_size=6)

# ---------------------------------------------------------------------------
# class_minimal: adaptive memory family (defaults of the paper runner)
# ---------------------------------------------------------------------------
M_CLASSES = ["AdaptiveMemoryFSGD", "AdaptiveMemoryFRMSprop", "AdaptiveMemoryFAdam",
             "AdaptiveMemoryFAdadelta", "AdaptiveMemoryFAdagrad", "AdaptiveMemoryFAdamW",
             "AdaptiveMemoryFNadam", "AdaptiveMemoryFAdamax"]
M_LR = {"AdaptiveMemoryFSGD": 5e-2, "AdaptiveMemoryFRMSprop": 1e-2, "AdaptiveMemoryFAdam": 1e-2,
        "AdaptiveMemoryFAdadelta": 1.0, "AdaptiveMemoryFAdagrad": 1e-1, "AdaptiveMemoryFAdamW": 1e-2,
        "AdaptiveMemoryFNadam": 1e-2, "AdaptiveMemoryFAdamax": 1e-2}
PAPER_ADAPTIVE = dict(history_size=4, normalize_coefficients=True, initial_mix=0.10,
                      min_mix=0.0, max_mix=0.30, adapt_step=0.01, mix_ema_gamma=0.90,
                      stability_ema_gamma=0.95, stability_threshold=0.25, stability_clip=10.0,
                      norm_match_fractional_gradient=True, use_raw_gradient_for_second_moment=True,
                      loss_ema_beta=0.95, loss_worsen_tolerance=1e-4, loss_mix_decrease=0.02,
                      warmup_steps=0)
for cls in M_CLASSES:
    for nu in (0.75, 1.25, 1.5):
        add("M", cls, f"paper,nu={nu}", lr=M_LR[cls], vderiv=nu, **PAPER_ADAPTIVE)
    add("M", cls, "nu=1(classical)", lr=M_LR[cls], vderiv=1.0, history_size=4)
    add("M", cls, "nu=0.9,K=6,raw-off,normmatch-off,descent-off,warmup=10,threshold=1.0",
        lr=M_LR[cls], vderiv=0.9, history_size=6, normalize_coefficients=False,
        use_raw_gradient_for_second_moment=False, norm_match_fractional_gradient=False,
        descent_check=False, warmup_steps=10, stability_threshold=1.0, max_mix=0.5,
        adapt_step=0.05)
    # Loss feedback through set_current_loss(): only comparable eagerly,
    # because the legacy file stores the flag as a Python float that a
    # tf.function freezes at trace time.
    add("M", cls, "nu=0.75,loss-feedback", modes=("eager",), loss_feedback=True,
        lr=M_LR[cls], vderiv=0.75, **dict(PAPER_ADAPTIVE, loss_mix_decrease=0.05,
                                          loss_worsen_tolerance=0.0))
add("M", "AdaptiveMemoryFSGD", "paper,nu=1.25,mom", lr=5e-2, momentum=0.9, vderiv=1.25, **PAPER_ADAPTIVE)
add("M", "AdaptiveMemoryFSGD", "paper,nu=1.25,mom,nesterov", lr=5e-2, momentum=0.9, nesterov=True, vderiv=1.25, **PAPER_ADAPTIVE)
add("M", "AdaptiveMemoryFRMSprop", "paper,nu=0.75,mom,centered", lr=1e-2, momentum=0.5, centered=True, vderiv=0.75, **PAPER_ADAPTIVE)
add("M", "AdaptiveMemoryFAdam", "paper,nu=0.75,amsgrad", lr=1e-2, amsgrad=True, vderiv=0.75, **PAPER_ADAPTIVE)

# ---------------------------------------------------------------------------
# class_d: related-work optimizers
# ---------------------------------------------------------------------------
for nu in (0.75, 1.25, 1.5):
    add("D", "AdaGL", f"alpha={nu},K=10", lr=1e-2, alpha=nu, history_size=10)
    add("D", "FCSGD_GL", f"alpha={nu},K=11,p=1", lr=5e-2, alpha=nu, history_size=11, perturbation_prob=1.0)
    add("D", "FCSGD_GL", f"alpha={nu},K=11,p=1,mom", lr=5e-2, momentum=0.9, alpha=nu, history_size=11, perturbation_prob=1.0)
    add("D", "FCAdam_GL", f"alpha={nu},K=11,p=1", lr=1e-2, alpha=nu, history_size=11, perturbation_prob=1.0)
    # Bernoulli mask p=0.5: same random stream only in eager mode.
    add("D", "FCSGD_GL", f"alpha={nu},K=11,p=0.5", modes=("eager",), lr=5e-2, alpha=nu, history_size=11, perturbation_prob=0.5)
    add("D", "FCAdam_GL", f"alpha={nu},K=11,p=0.5", modes=("eager",), lr=1e-2, alpha=nu, history_size=11, perturbation_prob=0.5)
    amax = max(1.5, nu)
    add("D", "AOFGD_SGD", f"alpha_init={nu}", lr=5e-2, alpha_init=nu, alpha_min=0.5, alpha_max=amax,
        delta_alpha=0.02, cef_threshold=0.05, ema_gamma=0.95, frac_epsilon=1e-6, eps_cef=1e-8)
    add("D", "AOFGD_SGD", f"alpha_init={nu},mom", lr=5e-2, momentum=0.9, alpha_init=nu, alpha_min=0.5,
        alpha_max=amax)
    add("D", "AOFGD_Adam", f"alpha_init={nu}", lr=1e-2, alpha_init=nu, alpha_min=0.5, alpha_max=amax,
        delta_alpha=0.02, cef_threshold=0.05, ema_gamma=0.95, frac_epsilon=1e-6, eps_cef=1e-8)
add("D", "AdaGL", "alpha=0.9,K=10,amsgrad", lr=1e-2, alpha=0.9, history_size=10, amsgrad=True)
add("D", "AdaGL", "alpha=1(classical)", lr=1e-2, alpha=1.0, history_size=10)
add("D", "FCSGD_GL", "alpha=0.9,K=11,p=1,mom,nesterov", lr=5e-2, momentum=0.9, nesterov=True, alpha=0.9, history_size=11, perturbation_prob=1.0)
add("D", "FCAdam_GL", "alpha=0.9,K=11,p=1,amsgrad", lr=1e-2, amsgrad=True, alpha=0.9, history_size=11, perturbation_prob=1.0)
add("D", "AOFGD_Adam", "alpha_init=0.9,amsgrad", lr=1e-2, amsgrad=True, alpha_init=0.9, alpha_min=0.5, alpha_max=1.9)


# =============================================================================
# STEP 4  RECORDED GRADIENT SEQUENCE (OPEN LOOP)
# =============================================================================
# 60 gradients and losses of plain float64 SGD (lr 0.05) on the toy problem.
# In the open loop both optimizers receive exactly this sequence.

recorded_weights = [w.astype(np.float64) for w in INITIAL_WEIGHTS]
RECORDED_LOSSES, RECORDED_GRADIENTS = [], []
for t in range(N_STEPS):
    x_batch = X[(t % 4) * 16:(t % 4) * 16 + 16].astype(np.float64)
    y_batch = Y[(t % 4) * 16:(t % 4) * 16 + 16]
    loss, gradients = loss_and_gradients(recorded_weights, x_batch, y_batch, np.float64)
    RECORDED_LOSSES.append(loss)
    RECORDED_GRADIENTS.append(gradients)
    recorded_weights = [w - 0.05 * g for w, g in zip(recorded_weights, gradients)]


# =============================================================================
# STEP 5  ONE TRAINING RUN IN KERAS 3 AND ONE IN PYTORCH
# =============================================================================
# Both functions do the same thing: create the optimizer from the
# configuration, run N_STEPS steps (closed or open loop), call
# set_current_loss() after every step for the loss-feedback runs, and return
# the final weights, the loss curve, lambda_t / alpha_t if the optimizer has
# them, and how often a random mask was drawn.
#
# The random masks of FCSGD_GL / FCAdam_GL: tf.random.uniform and torch.rand
# are replaced, for the duration of one run, by the same seeded NumPy stream.

_original_tf_cast = tf.cast
_original_tf_uniform = tf.random.uniform
_original_torch_rand = torch.rand


def _tf_cast_python_numbers_exactly(x, dtype, name=None):
    """tf.cast without the float32 detour for Python numbers (float64 runs only)."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return tf.constant(x, dtype=dtype, name=name)
    return _original_tf_cast(x, dtype, name=name)


def run_keras(configuration, dtype, loop):
    np_dtype = np.float64 if dtype == "float64" else np.float32
    mask_stream = np.random.default_rng(2024)
    mask_draws = [0]

    def shared_uniform(shape, dtype=tf.float32, **kwargs):
        mask_draws[0] += 1
        return tf.constant(mask_stream.random(shape[0]).astype(np.float32), dtype)

    if dtype == "float64":
        tf.cast = _tf_cast_python_numbers_exactly
    tf.random.uniform = shared_uniform
    try:
        arguments = dict(configuration["kwargs"])
        learning_rate = arguments.pop("lr")
        optimizer_class = getattr(KERAS_MODULES[configuration["module"]], configuration["cls"])
        optimizer = optimizer_class(learning_rate=learning_rate, **arguments)
        variables = [keras.Variable(w.astype(np_dtype), dtype=dtype) for w in INITIAL_WEIGHTS]
        losses = []
        for t in range(N_STEPS):
            if loop == "closed":
                x_batch = X[(t % 4) * 16:(t % 4) * 16 + 16].astype(np_dtype)
                y_batch = Y[(t % 4) * 16:(t % 4) * 16 + 16]
                loss, gradients = loss_and_gradients([v.numpy() for v in variables], x_batch, y_batch, np_dtype)
            else:
                loss, gradients = RECORDED_LOSSES[t], [g.astype(np_dtype) for g in RECORDED_GRADIENTS[t]]
            optimizer.apply_gradients(zip([tf.constant(g) for g in gradients], variables))
            if configuration["loss_feedback"]:
                optimizer.set_current_loss(loss)
            losses.append(loss)
    finally:
        tf.cast = _original_tf_cast
        tf.random.uniform = _original_tf_uniform
    result = {"w": [v.numpy().astype(np.float64) for v in variables], "loss": np.array(losses),
              "mask_draws": mask_draws[0]}
    if hasattr(optimizer, "current_mix_values"):
        result["mix"] = np.array([float(v) for v in optimizer.current_mix_values()])
    if hasattr(optimizer, "current_alpha_values"):
        result["alpha"] = np.array([float(v) for v in optimizer.current_alpha_values()])
    return result


def run_torch(configuration, dtype, loop):
    np_dtype = np.float64 if dtype == "float64" else np.float32
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    mask_stream = np.random.default_rng(2024)
    mask_draws = [0]

    def shared_rand(n, **kwargs):
        mask_draws[0] += 1
        return torch.from_numpy(mask_stream.random(n).astype(np.float32))

    torch.rand = shared_rand
    try:
        arguments = dict(configuration["kwargs"])
        learning_rate = float(np.float32(arguments.pop("lr")))     # Keras' lr variable is float32
        optimizer_class = getattr(TORCH_MODULES[configuration["module"]], configuration["cls"])
        parameters = [torch.nn.Parameter(torch.tensor(w.astype(np_dtype), dtype=torch_dtype))
                      for w in INITIAL_WEIGHTS]
        optimizer = optimizer_class(parameters, lr=learning_rate, **arguments)
        losses = []
        for t in range(N_STEPS):
            if loop == "closed":
                x_batch = X[(t % 4) * 16:(t % 4) * 16 + 16].astype(np_dtype)
                y_batch = Y[(t % 4) * 16:(t % 4) * 16 + 16]
                loss, gradients = loss_and_gradients([p.detach().numpy() for p in parameters],
                                                     x_batch, y_batch, np_dtype)
            else:
                loss, gradients = RECORDED_LOSSES[t], [g.astype(np_dtype) for g in RECORDED_GRADIENTS[t]]
            for p, g in zip(parameters, gradients):
                p.grad = torch.from_numpy(g.copy())
            optimizer.step()
            if configuration["loss_feedback"]:
                optimizer.set_current_loss(loss)
            losses.append(loss)
    finally:
        torch.rand = _original_torch_rand
    result = {"w": [p.detach().numpy().astype(np.float64) for p in parameters], "loss": np.array(losses),
              "mask_draws": mask_draws[0]}
    if hasattr(optimizer, "current_mix_values"):
        result["mix"] = np.array([float(v) for v in optimizer.current_mix_values()])
    if hasattr(optimizer, "current_alpha_values"):
        result["alpha"] = np.array([float(v) for v in optimizer.current_alpha_values()])
    return result


# =============================================================================
# STEP 6  RUN EVERY CONFIGURATION IN BOTH FRAMEWORKS
# =============================================================================

results = {}          # results[(framework, dtype, loop)][configuration id]
if RUN_OPTIMIZER_CHECK:
    runs = []
    for loop in ("open", "closed"):
        for dtype in DTYPES:
            runs += [("keras3", dtype, loop), ("torch", dtype, loop)]
        if "float32" in DTYPES and "float64" not in DTYPES:
            runs.append(("keras3", "float64", loop))          # reference for the float32 criterion
    for framework, dtype, loop in runs:
        run_start = time.time()
        results[(framework, dtype, loop)] = {}
        for configuration in CONFIGS:
            if framework == "keras3":
                results[(framework, dtype, loop)][configuration["id"]] = run_keras(configuration, dtype, loop)
            else:
                results[(framework, dtype, loop)][configuration["id"]] = run_torch(configuration, dtype, loop)
        print(f"[STEP 6] {framework:6s} {dtype} {loop:6s} loop: {len(CONFIGS)} configurations "
              f"in {time.time() - run_start:.0f} s")


# =============================================================================
# STEP 7  COMPARE THE OPTIMIZERS
# =============================================================================

def relative_difference(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))


def all_differences(result_a, result_b):
    """Largest relative difference of the weights, of lambda_t/alpha_t, and of the loss curve."""
    differences = {"w": max(relative_difference(a, b) for a, b in zip(result_a["w"], result_b["w"])), "d": 0.0}
    for key in ("mix", "alpha"):
        if key in result_a or key in result_b:
            differences["d"] = max(differences["d"], relative_difference(result_a[key], result_b[key]))
    differences["loss"] = float(np.max(np.abs(result_a["loss"] - result_b["loss"]) / np.abs(result_a["loss"])))
    return differences


optimizer_summary = []
if RUN_OPTIMIZER_CHECK:
    for loop in ("open", "closed"):
        for dtype in DTYPES:
            keras_results = results[("keras3", dtype, loop)]
            torch_results = results[("torch", dtype, loop)]
            reference = results[("keras3", "float64", loop)] if dtype == "float32" else None
            if dtype == "float64":
                criterion = f"torch-vs-keras <= {TOLERANCE_FLOAT64[loop]:g} relative"
            else:
                criterion = f"torch-vs-keras <= max({FLOAT32_FACTOR:g} x keras32-vs-keras64, {FLOAT32_FLOOR:g})"
            print(f"\n[STEP 7] {loop.upper()} LOOP, {dtype}, {N_STEPS} steps, pass if {criterion}")
            if PRINT_EVERY_CONFIGURATION:
                print(f"   {'family':6s} {'configuration':70s} {'weights':>8s} {'lam/alp':>8s} {'loss':>8s}")
            rows = []
            for configuration in CONFIGS:
                cid = configuration["id"]
                differences = all_differences(keras_results[cid], torch_results[cid])
                draws_keras, draws_torch = keras_results[cid]["mask_draws"], torch_results[cid]["mask_draws"]
                random_ok = draws_keras == draws_torch
                if configuration["kwargs"].get("perturbation_prob", 1.0) not in (0.0, 1.0):
                    random_ok = random_ok and draws_keras == N_STEPS * N_VARIABLES   # one mask per variable and step
                finite = all(np.isfinite(w).all() for w in torch_results[cid]["w"])
                if dtype == "float64":
                    ok = all(differences[k] <= TOLERANCE_FLOAT64[loop] for k in ("w", "d", "loss"))
                else:
                    own_error = all_differences(reference[cid], keras_results[cid])   # Keras float32 vs float64
                    ok = all(differences[k] <= max(FLOAT32_FACTOR * own_error[k], FLOAT32_FLOOR)
                             for k in ("w", "d", "loss"))
                ok = ok and random_ok and finite
                rows.append((configuration["module"], cid, differences, ok))
                if PRINT_EVERY_CONFIGURATION:
                    print(f"   {configuration['module']:6s} {cid[:70]:70s} {differences['w']:8.1e} "
                          f"{differences['d']:8.1e} {differences['loss']:8.1e}  {'PASS' if ok else 'FAIL'}")
            for family, family_name in (("B", "Herrera"), ("C", "GL memory"), ("M", "adaptive memory"),
                                        ("D", "related work")):
                family_rows = [r for r in rows if r[0] == family]
                print(f"   {family_name:16s} {sum(r[3] for r in family_rows):3d}/{len(family_rows):3d} PASS, "
                      f"worst weights {max(r[2]['w'] for r in family_rows):.1e}, "
                      f"worst lambda/alpha {max(r[2]['d'] for r in family_rows):.1e}, "
                      f"worst loss {max(r[2]['loss'] for r in family_rows):.1e}")
            n_pass = sum(r[3] for r in rows)
            print(f"   {loop} loop, {dtype}: {n_pass}/{len(rows)} PASS")
            for r in rows:
                if not r[3]:
                    print("   FAIL", r)
            optimizer_summary.append((loop, dtype, n_pass, len(rows)))


# =============================================================================
# STEP 8  COMPARE THE FRACTAL ACTIVATIONS
# =============================================================================
# The random draw of the depth is framework-specific (tf.random vs
# torch.randint), so the depth is fixed by replacing _sample_random_depth in
# both modules (minimum, middle and maximum of the depth range).

activation_summary = []
if RUN_ACTIVATION_CHECK:
    print("\n[STEP 8] fractal activations, values and gradients")
    grid = np.linspace(-5.0, 5.0, 20001)
    batch = np.random.default_rng(3).normal(scale=2.0, size=(64, 32))
    FLOAT32_INTERNAL = {"modulated_blancmange_curve", "decaying_cosine_function_tf",
                        "cantor_function_tf", "cantor_set_activation_tf", "random_depth_cantor_function_tf"}
    ALWAYS_FLOAT32 = {"random_depth_modulated_blancmange_curve", "random_depth_decaying_cosine_function_tf"}
    original_keras_depth = faf_keras._sample_random_depth
    original_torch_depth = faf_torch._sample_random_depth

    print(f"   {'activation':55s} {'input':8s} {'values':>9s} {'grads':>9s} {'tol':>7s} {'grad>1e-3':>9s}")
    for name in faf_keras.ALL_FRACTAL_ACTIVATIONS:
        keras_fn = faf_keras.ALL_FRACTAL_ACTIVATIONS[name]
        torch_fn = faf_torch.ALL_FRACTAL_ACTIVATIONS[name]
        for dtype in ("float32", "float64"):
            if name in ALWAYS_FLOAT32 or (dtype == "float32" and name in FLOAT32_INTERNAL):
                tolerance = 2e-5
            else:
                tolerance = 1e-12 if dtype == "float64" else 1e-6
            worst_value, worst_gradient, worst_outliers, ok = 0.0, 0.0, 0.0, True
            fractions = (0.0, 0.5, 1.0) if name.startswith("random_depth") else (None,)
            for fraction in fractions:
                if fraction is not None:
                    faf_keras._sample_random_depth = lambda min_terms=10, max_terms=50, f=fraction: tf.constant(
                        int(round(min_terms + f * (max_terms - min_terms))), tf.int32)
                    faf_torch._sample_random_depth = lambda min_terms=10, max_terms=50, device=None, f=fraction: \
                        torch.tensor(int(round(min_terms + f * (max_terms - min_terms))), device=device)
                for data in (grid, batch):
                    # Keras: value and gradient with GradientTape
                    x_keras = tf.constant(data.astype(dtype))
                    with tf.GradientTape() as tape:
                        tape.watch(x_keras)
                        y_keras_tensor = keras_fn(x_keras)
                    g_keras_tensor = tape.gradient(y_keras_tensor, x_keras)
                    y_keras = y_keras_tensor.numpy()
                    g_keras = None if g_keras_tensor is None else g_keras_tensor.numpy()
                    # PyTorch: value and gradient with backward()
                    x_torch = torch.tensor(data.astype(dtype), requires_grad=True)
                    y_torch_tensor = torch_fn(x_torch)
                    g_torch = None
                    if y_torch_tensor.requires_grad:
                        y_torch_tensor.sum().backward()
                        g_torch = x_torch.grad.numpy()
                    y_torch = y_torch_tensor.detach().numpy()
                    # identical structure: dtype, gradient path, NaN/inf positions
                    ok &= y_keras.dtype == y_torch.dtype and (g_keras is None) == (g_torch is None)
                    ok &= np.array_equal(np.isnan(y_keras), np.isnan(y_torch))
                    ok &= np.array_equal(np.isinf(y_keras), np.isinf(y_torch))
                    finite = np.isfinite(y_keras)
                    value_difference = 0.0
                    if finite.any():
                        a, b = y_keras[finite].astype(np.float64), y_torch[finite].astype(np.float64)
                        value_difference = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))
                    gradient_difference, outlier_share = 0.0, 0.0
                    if g_keras is not None and g_torch is not None:
                        ok &= np.array_equal(np.isnan(g_keras), np.isnan(g_torch))
                        ok &= np.array_equal(np.isinf(g_keras), np.isinf(g_torch))
                        finite = np.isfinite(g_keras)
                        if finite.any():
                            a, b = g_keras[finite].astype(np.float64), g_torch[finite].astype(np.float64)
                            scale = max(np.max(np.abs(a)), 1e-300)
                            gradient_difference = float(np.max(np.abs(a - b)) / scale)
                            outlier_share = float(np.mean(np.abs(a - b) > 1e-3 * scale))
                    if name == "random_depth_cantor_function_tf" and dtype == "float32" and fraction == 1.0:
                        ok &= value_difference <= 1e-4 and outlier_share <= 0.005     # documented exception
                    else:
                        ok &= value_difference <= tolerance and gradient_difference <= tolerance
                    worst_value = max(worst_value, value_difference)
                    worst_gradient = max(worst_gradient, gradient_difference)
                    worst_outliers = max(worst_outliers, outlier_share)
            faf_keras._sample_random_depth = original_keras_depth
            faf_torch._sample_random_depth = original_torch_depth
            activation_summary.append((name, dtype, bool(ok)))
            print(f"   {name:55s} {dtype:8s} {worst_value:9.1e} {worst_gradient:9.1e} {tolerance:7.0e} "
                  f"{100 * worst_outliers:8.2f}%  {'PASS' if ok else 'FAIL'}")
    print(f"   {sum(r[2] for r in activation_summary)}/{len(activation_summary)} activation x dtype "
          f"combinations PASS")


# =============================================================================
# STEP 9  SUMMARY
# =============================================================================

print(f"\nfinished in {time.time() - start_time:.0f} s")
all_ok = True
for loop, dtype, n_pass, n_total in optimizer_summary:
    print(f"   optimizers, {loop:6s} loop, {dtype}: {n_pass}/{n_total} PASS")
    all_ok &= n_pass == n_total
if activation_summary:
    n_pass = sum(r[2] for r in activation_summary)
    print(f"   activations: {n_pass}/{len(activation_summary)} PASS")
    all_ok &= n_pass == len(activation_summary)
print("ALL PASSED" if all_ok else "FAILURES (see above)")
sys.exit(0 if all_ok else 1)
