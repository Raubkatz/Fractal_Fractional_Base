#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_tests_tensorflow_vs_legacy.py
================================

Checks that the Keras 3 files compute exactly what the original (legacy,
optimizer_v2 / TensorFlow <= 2.15) files compute: the same weights after
the same training steps, bit for bit, and the same activation values and
gradients, bit for bit. Written top to bottom, step by step.

STEP  0  Settings
STEP  1  Imports: the legacy files through tf_keras, the Keras 3 files through Keras 3
STEP  2  The toy problem: a 4-8-3 tanh network, loss and gradients in TensorFlow
STEP  3  The 167 optimizer configurations (the same list as in 05_tests_pytorch_vs_tensorflow.py)
STEP  4  One training run (one function, used for both implementations)
STEP  5  Run every configuration with both implementations and compare bit for bit
STEP  6  Compare the fractal activations bit for bit
STEP  7  Summary

Every configuration runs with the legacy class and with the Keras 3 class on
the same network, the same initial weights and the same mini-batches, for 60
steps, in eager mode, inside tf.function (graph mode), and inside
tf.function with TensorFlow's graph optimizer (Grappler) switched off. Pass:
the final weights and lambda_t / alpha_t are bit-identical in every run.
Fourteen configurations run only in eager mode, because they can only be
compared there: the loss-feedback runs (the legacy flag is a Python float that
tf.function freezes at trace time) and the FCGL runs with p = 0.5 (the random
stream depends on how the graph is built).

What you need
-------------
1. The legacy package that keeps the old optimizer API alive:
       pip install tf-keras==2.21.0
2. The original files of the paper code, in LEGACY_CODE_FOLDER (STEP 0;
   by default the folder of this script, their names do not clash):
       class_b_tf_fractional_optimizers_OHA.py
       class_c_tf_fractional_optimizers_SR.py
       class_minimal_fractional_adaptive_optimizers.py
       class_d_related_work_optimizers.py
       fractal_activation_functions.py

Run it with
    python 06_tests_tensorflow_vs_legacy.py

(no command-line arguments: every setting is in STEP 0). Takes about 15
minutes on a 2-core CPU.
"""

# =============================================================================
# STEP 0  SETTINGS
# =============================================================================

LEGACY_CODE_FOLDER = "."              # folder with the original (legacy) files
RUN_OPTIMIZER_CHECK = True
RUN_ACTIVATION_CHECK = True
MODES = ("eager", "graph", "graph_without_grappler")
N_STEPS = 60                          # training steps per configuration
PRINT_EVERY_CONFIGURATION = False     # True: one line per configuration and mode


# =============================================================================
# STEP 1  IMPORTS
# =============================================================================
# The legacy files subclass tf.keras.optimizers.Optimizer and use the old
# optimizer_v2 API. With TF_USE_LEGACY_KERAS=1 (set BEFORE TensorFlow is
# imported) tf.keras is the tf_keras package, and its old base class is
# tf.keras.optimizers.legacy.Optimizer. The Keras 3 files import the keras
# package directly, so both run side by side in this process.

import os
import sys
import time

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

tf.keras.optimizers.Optimizer = tf.keras.optimizers.legacy.Optimizer   # for class_b_... and class_d_...

LEGACY_FILES = ["class_b_tf_fractional_optimizers_OHA.py", "class_c_tf_fractional_optimizers_SR.py",
                "class_minimal_fractional_adaptive_optimizers.py", "class_d_related_work_optimizers.py",
                "fractal_activation_functions.py"]
missing = [f for f in LEGACY_FILES if not os.path.exists(os.path.join(LEGACY_CODE_FOLDER, f))]
if missing:
    sys.exit(f"The original files {missing} are not in LEGACY_CODE_FOLDER = {LEGACY_CODE_FOLDER!r} "
             f"(see STEP 0).")
sys.path.insert(0, os.path.abspath(LEGACY_CODE_FOLDER))

import class_b_tf_fractional_optimizers_OHA
import class_c_tf_fractional_optimizers_SR
import class_minimal_fractional_adaptive_optimizers
import class_d_related_work_optimizers
import fractal_activation_functions as faf_legacy

import keras
import class_b_tf_fractional_optimizers_OHA_keras3
import class_c_tf_fractional_optimizers_SR_keras3
import class_minimal_fractional_adaptive_optimizers_keras3
import class_d_related_work_optimizers_keras3
import fractal_activation_functions_keras3 as faf_keras

LEGACY_MODULES = {
    "B": class_b_tf_fractional_optimizers_OHA,
    "C": class_c_tf_fractional_optimizers_SR,
    "M": class_minimal_fractional_adaptive_optimizers,
    "D": class_d_related_work_optimizers,
}
KERAS3_MODULES = {
    "B": class_b_tf_fractional_optimizers_OHA_keras3,
    "C": class_c_tf_fractional_optimizers_SR_keras3,
    "M": class_minimal_fractional_adaptive_optimizers_keras3,
    "D": class_d_related_work_optimizers_keras3,
}
start_time = time.time()
print(f"TensorFlow {tf.__version__}; legacy files through tf.keras = {tf.keras.__name__}; "
      f"Keras 3 files through keras {keras.__version__}")


# =============================================================================
# STEP 2  THE TOY PROBLEM
# =============================================================================
# 64 samples, a 4-8-3 tanh network, softmax cross-entropy, mini-batches of 16
# cycled in a fixed order. Loss and gradients use the same TensorFlow ops for
# both implementations, so any difference comes from the optimizers.

rng = np.random.default_rng(0)
X = rng.normal(size=(64, 4)).astype(np.float32)
Y = rng.integers(0, 3, size=(64,)).astype(np.int32)
INITIAL_WEIGHTS = [
    (rng.normal(size=(4, 8)) * 0.7).astype(np.float32),
    (rng.normal(size=(8,)) * 0.1).astype(np.float32),
    (rng.normal(size=(8, 3)) * 0.7).astype(np.float32),
    (rng.normal(size=(3,)) * 0.1).astype(np.float32),
]
BATCHES = [(X[i:i + 16], Y[i:i + 16]) for i in range(0, 64, 16)]


def loss_and_gradients(variables, x_batch, y_batch):
    weights = [tf.convert_to_tensor(v) for v in variables]
    with tf.GradientTape() as tape:
        tape.watch(weights)
        h = tf.tanh(tf.matmul(x_batch, weights[0]) + weights[1])
        logits = tf.matmul(h, weights[2]) + weights[3]
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch, logits=logits))
    return loss, tape.gradient(loss, weights)


# =============================================================================
# STEP 3  THE 167 OPTIMIZER CONFIGURATIONS
# =============================================================================
# Every class; nu < 1, nu = 1 (classical limit) and nu > 1; several history
# lengths; momentum, Nesterov, centered and AMSGrad variants; the memory
# switches; warm-up; loss feedback; Bernoulli masks with p = 1 and p = 0.5.
# "modes": configurations that can only be compared in eager mode (see top).

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
# STEP 4  ONE TRAINING RUN
# =============================================================================
# implementation = "legacy" (original file, lr=) or "keras3" (new file,
# learning_rate=); mode = "eager", "graph" (tf.function) or
# "graph_without_grappler". Returns the final weights and lambda_t / alpha_t.

def run(implementation, configuration, mode):
    tf.random.set_seed(1234)
    arguments = dict(configuration["kwargs"])
    learning_rate = arguments.pop("lr")
    if implementation == "legacy":
        variables = [tf.Variable(w) for w in INITIAL_WEIGHTS]
        optimizer_class = getattr(LEGACY_MODULES[configuration["module"]], configuration["cls"])
        optimizer = optimizer_class(lr=learning_rate, **arguments)
    else:
        variables = [keras.Variable(w) for w in INITIAL_WEIGHTS]
        optimizer_class = getattr(KERAS3_MODULES[configuration["module"]], configuration["cls"])
        optimizer = optimizer_class(learning_rate=learning_rate, **arguments)

    def training_step(x_batch, y_batch):
        loss, gradients = loss_and_gradients(variables, x_batch, y_batch)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    if mode != "eager":
        training_step = tf.function(training_step)
    for t in range(N_STEPS):
        x_batch, y_batch = BATCHES[t % len(BATCHES)]
        loss = training_step(tf.constant(x_batch), tf.constant(y_batch))
        if configuration["loss_feedback"]:
            optimizer.set_current_loss(float(loss))
    result = {f"w{i}": np.array(v.numpy()) for i, v in enumerate(variables)}
    if hasattr(optimizer, "current_mix_values"):
        result["mix"] = np.array([float(v) for v in optimizer.current_mix_values()])
    if hasattr(optimizer, "current_alpha_values"):
        result["alpha"] = np.array([float(v) for v in optimizer.current_alpha_values()])
    return result


# =============================================================================
# STEP 5  RUN EVERY CONFIGURATION WITH BOTH IMPLEMENTATIONS
# =============================================================================

optimizer_rows = []          # (mode, family, configuration id, max |difference|, bit-identical)
if RUN_OPTIMIZER_CHECK:
    for mode in MODES:
        mode_start = time.time()
        # Grappler (TensorFlow's graph optimizer) on or off for every graph built from here on
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": mode == "graph_without_grappler"})
        configurations = [c for c in CONFIGS if mode == "eager" or "graph" in c["modes"]]
        for configuration in configurations:
            legacy_result = run("legacy", configuration, mode)
            keras3_result = run("keras3", configuration, mode)
            same_keys = set(legacy_result) == set(keras3_result)
            identical = same_keys and all(np.array_equal(legacy_result[k], keras3_result[k]) for k in legacy_result)
            largest = max(float(np.max(np.abs(legacy_result[k].astype(np.float64) - keras3_result[k])))
                          if legacy_result[k].size else 0.0 for k in legacy_result) if same_keys else float("inf")
            optimizer_rows.append((mode, configuration["module"], configuration["id"], largest, identical))
            if PRINT_EVERY_CONFIGURATION or not identical:
                print(f"   {mode:22s} {configuration['id'][:70]:70s} max|diff| {largest:.3e}  "
                      f"{'identical' if identical else 'DIFFERENT'}")
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": False})
        mode_rows = [r for r in optimizer_rows if r[0] == mode]
        print(f"[STEP 5] {mode:22s} {sum(r[4] for r in mode_rows)}/{len(mode_rows)} configurations bit-identical "
              f"({time.time() - mode_start:.0f} s)")

    print(f"\n   {'family':34s} " + " ".join(f"{m:>24s}" for m in MODES))
    for family, family_name in (("B", "Herrera (class_b)"), ("C", "GL memory (class_c)"),
                                ("M", "adaptive memory (class_minimal)"), ("D", "related work (class_d)")):
        cells = []
        for mode in MODES:
            rows = [r for r in optimizer_rows if r[0] == mode and r[1] == family]
            cells.append(f"{sum(r[4] for r in rows)}/{len(rows)} identical")
        print(f"   {family_name:34s} " + " ".join(f"{c:>24s}" for c in cells))


# =============================================================================
# STEP 6  COMPARE THE FRACTAL ACTIVATIONS
# =============================================================================
# 20,001 points in [-5, 5] and a random 64 x 32 batch, in eager mode and in
# tf.function. The random-depth functions are compared with the same TF seed.

activation_rows = []         # (name, largest value difference, largest gradient difference, bit-identical)
if RUN_ACTIVATION_CHECK:
    print("\n[STEP 6] fractal activations, legacy vs Keras 3")
    grid = np.linspace(-5.0, 5.0, 20001).astype(np.float32)
    batch = np.random.default_rng(3).normal(scale=2.0, size=(64, 32)).astype(np.float32)
    for name, keras3_fn in faf_keras.ALL_FRACTAL_ACTIVATIONS.items():
        legacy_fn = getattr(faf_legacy, name)
        worst_value, worst_gradient, identical = 0.0, 0.0, True
        for mode in ("eager", "graph"):
            for data in (grid, batch):
                outputs = []
                for fn in (legacy_fn, keras3_fn):
                    def value_and_gradient(x, fn=fn):
                        with tf.GradientTape() as tape:
                            tape.watch(x)
                            y = fn(x)
                        return y, tape.gradient(y, x)
                    if mode == "graph":
                        value_and_gradient = tf.function(value_and_gradient)
                    tf.random.set_seed(7)
                    y, g = value_and_gradient(tf.constant(data))
                    outputs.append((y.numpy(), None if g is None else g.numpy()))
                (y_legacy, g_legacy), (y_keras3, g_keras3) = outputs
                identical &= y_legacy.dtype == y_keras3.dtype == np.float32
                identical &= bool(np.array_equal(y_legacy, y_keras3, equal_nan=True))
                worst_value = max(worst_value, float(np.nanmax(np.abs(y_legacy.astype(np.float64) - y_keras3))))
                if g_legacy is None or g_keras3 is None:
                    identical &= (g_legacy is None) == (g_keras3 is None)
                else:
                    identical &= bool(np.array_equal(g_legacy, g_keras3, equal_nan=True))
                    finite = np.isfinite(g_legacy) & np.isfinite(g_keras3)
                    if finite.any():
                        worst_gradient = max(worst_gradient, float(np.max(
                            np.abs(g_legacy[finite].astype(np.float64) - g_keras3[finite]))))
        activation_rows.append((name, worst_value, worst_gradient, bool(identical)))
        print(f"   {name:55s} max|dy| {worst_value:.3e}  max|dgrad| {worst_gradient:.3e}  "
              f"{'bit-identical' if identical else 'DIFFERENT'}")


# =============================================================================
# STEP 7  SUMMARY
# =============================================================================

print(f"\nfinished in {time.time() - start_time:.0f} s")
all_ok = True
if optimizer_rows:
    n_identical = sum(r[4] for r in optimizer_rows)
    print(f"   optimizers: {n_identical}/{len(optimizer_rows)} runs bit-identical "
          f"({len(CONFIGS)} configurations, modes {', '.join(MODES)})")
    all_ok &= n_identical == len(optimizer_rows)
if activation_rows:
    n_identical = sum(r[3] for r in activation_rows)
    print(f"   activations: {n_identical}/{len(activation_rows)} bit-identical (values and gradients)")
    all_ok &= n_identical == len(activation_rows)
print("ALL PASSED" if all_ok else "FAILURES (see above)")
sys.exit(0 if all_ok else 1)
