#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_tests_tensorflow.py
======================

Integration test of the TensorFlow / Keras 3 files: every optimizer and
every activation inside normal Keras 3 workflows. Written top to bottom,
step by step, like the hello-world scripts.

STEP  0  Settings
STEP  1  Imports
STEP  2  A small Iris-shaped data set
STEP  3  The 27 fractional optimizers
STEP  4  model.fit with the paper's BatchNorm/Dropout network, plain and with XLA
STEP  5  get_config() -> from_config() round trip
STEP  6  model.save(".keras") -> keras.models.load_model(): same predictions,
         same optimizer class, config and variables; training continues
STEP  7  learning-rate schedule + clipnorm (Keras 3 base arguments)
STEP  8  activations: Dense(activation=fn) and FractalActivation layer, save/load
STEP  9  all 18 activations under XLA (jit_compile=True)
STEP 10  memory stays flat when training many models in one process
STEP 11  Summary

Run it with
    python 03_tests_tensorflow.py

(no command-line arguments: every setting is in STEP 0). Needs the Keras 3
files in the same folder. Takes about 15 minutes on a 2-core CPU and about
5 GB of memory (TensorFlow keeps the compiled graphs of the ~90 models).
"""

# =============================================================================
# STEP 0  SETTINGS
# =============================================================================

EPOCHS_FIT = 8                 # STEP 4: epochs per optimizer
RUN_XLA_ACTIVATION_CHECK = True  # STEP 9
RUN_MEMORY_CHECK = True        # STEP 10
MEMORY_LIMIT_MB_PER_MODEL = 40  # STEP 10: allowed growth per model (Keras' own Adam: ~130 MB)


# =============================================================================
# STEP 1  IMPORTS
# =============================================================================

import gc
import os
import sys
import tempfile
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import keras

import class_b_tf_fractional_optimizers_OHA_keras3 as B
import class_c_tf_fractional_optimizers_SR_keras3 as C
import class_minimal_fractional_adaptive_optimizers_keras3 as M
import class_d_related_work_optimizers_keras3 as D
import fractal_activation_functions_keras3 as faf

start_time = time.time()
temporary_folder = tempfile.mkdtemp(prefix="03_tests_tensorflow_")
failures = []


def check(condition, message):
    """Record a failed check (the script keeps going and lists all failures at the end)."""
    if not condition:
        failures.append(message)
        print("   FAILED:", message)
    return bool(condition)


# =============================================================================
# STEP 2  DATA
# =============================================================================
# 150 samples, 4 features, 3 classes (the shape of Iris), fixed seed.

rng = np.random.default_rng(0)
X = rng.uniform(size=(150, 4)).astype(np.float32)
Y = (X[:, 0] * 2 + X[:, 2]).astype(np.int32) % 3


# =============================================================================
# STEP 3  THE 27 FRACTIONAL OPTIMIZERS
# =============================================================================
# (name, function that creates a NEW optimizer)

OPTIMIZERS = [
    ("FSGD", lambda: B.FSGD(learning_rate=1e-2, momentum=0.9, vderiv=0.75)),
    ("FAdam", lambda: B.FAdam(learning_rate=1e-2, vderiv=1.25)),
    ("FAdagrad", lambda: B.FAdagrad(learning_rate=5e-2, vderiv=0.75)),
    ("FAdadelta", lambda: B.FAdadelta(learning_rate=1.0, vderiv=1.25)),
    ("FRMSprop", lambda: B.FRMSprop(learning_rate=1e-2, vderiv=0.75)),
    ("FAdamW", lambda: B.FAdamW(learning_rate=1e-2, weight_decay=0.004, vderiv=0.75)),
    ("MemoryFSGD", lambda: C.MemoryFSGD(learning_rate=1e-2, momentum=0.9, vderiv=0.75, history_size=6)),
    ("MemoryFRMSprop", lambda: C.MemoryFRMSprop(learning_rate=1e-2, vderiv=1.25, history_size=6)),
    ("MemoryFAdam", lambda: C.MemoryFAdam(learning_rate=1e-2, vderiv=1.5, history_size=6)),
    ("MemoryFAdadelta", lambda: C.MemoryFAdadelta(learning_rate=1.0, vderiv=0.75, history_size=6)),
    ("MemoryFAdagrad", lambda: C.MemoryFAdagrad(learning_rate=5e-2, vderiv=0.75, history_size=6)),
    ("MemoryFAdamW", lambda: C.MemoryFAdamW(learning_rate=1e-2, weight_decay=0.004, vderiv=0.75, history_size=6)),
    ("MemoryFNadam", lambda: C.MemoryFNadam(learning_rate=1e-2, vderiv=0.75, history_size=6)),
    ("MemoryFAdamax", lambda: C.MemoryFAdamax(learning_rate=1e-2, vderiv=0.75, history_size=6)),
    ("AdaptiveMemoryFSGD", lambda: M.AdaptiveMemoryFSGD(learning_rate=1e-2, momentum=0.9, vderiv=1.25)),
    ("AdaptiveMemoryFRMSprop", lambda: M.AdaptiveMemoryFRMSprop(learning_rate=1e-2, vderiv=1.25)),
    ("AdaptiveMemoryFAdam", lambda: M.AdaptiveMemoryFAdam(learning_rate=1e-2, vderiv=1.25)),
    ("AdaptiveMemoryFAdadelta", lambda: M.AdaptiveMemoryFAdadelta(learning_rate=1.0, vderiv=0.75)),
    ("AdaptiveMemoryFAdagrad", lambda: M.AdaptiveMemoryFAdagrad(learning_rate=5e-2, vderiv=0.75)),
    ("AdaptiveMemoryFAdamW", lambda: M.AdaptiveMemoryFAdamW(learning_rate=1e-2, weight_decay=0.004, vderiv=0.75)),
    ("AdaptiveMemoryFNadam", lambda: M.AdaptiveMemoryFNadam(learning_rate=1e-2, vderiv=0.75)),
    ("AdaptiveMemoryFAdamax", lambda: M.AdaptiveMemoryFAdamax(learning_rate=1e-2, vderiv=0.75)),
    ("AdaGL", lambda: D.AdaGL(learning_rate=1e-2, alpha=0.75, history_size=10)),
    ("FCSGD_GL", lambda: D.FCSGD_GL(learning_rate=1e-2, momentum=0.9, alpha=0.75, history_size=11)),
    ("FCAdam_GL", lambda: D.FCAdam_GL(learning_rate=1e-2, alpha=0.75, history_size=11)),
    ("AOFGD_SGD", lambda: D.AOFGD_SGD(learning_rate=1e-2, momentum=0.9, alpha_init=0.75, alpha_max=1.5)),
    ("AOFGD_Adam", lambda: D.AOFGD_Adam(learning_rate=1e-2, alpha_init=1.25, alpha_max=1.5)),
]


# =============================================================================
# STEP 4  model.fit, PLAIN AND WITH XLA
# =============================================================================
# The paper's network (Dense -> BN -> Dropout, twice, softmax). The loss must
# stay finite and decrease (noisy tiny problem: mean of the last 3 epochs
# below epoch 1). The XLA-trained model of every optimizer is kept for STEP 6.

print(f"[STEP 4] model.fit for {len(OPTIMIZERS)} optimizers, {EPOCHS_FIT} epochs, without and with XLA")
xla_models = {}
for name, make_optimizer in OPTIMIZERS:
    losses = {}
    for jit in (False, True):
        keras.utils.set_random_seed(1)
        initializer = keras.initializers.GlorotUniform(seed=42)
        model = keras.Sequential([
            keras.Input(shape=(4,)),
            keras.layers.Dense(16, activation="tanh", kernel_initializer=initializer),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="tanh", kernel_initializer=initializer),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(3, activation="softmax", kernel_initializer=initializer),
        ])
        model.compile(optimizer=make_optimizer(), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"], jit_compile=jit)
        history = model.fit(X, Y, epochs=EPOCHS_FIT, batch_size=16, verbose=0)
        losses[jit] = history.history["loss"]
        check(np.all(np.isfinite(losses[jit])), f"{name}: non-finite loss (jit={jit})")
        check(np.mean(losses[jit][-3:]) < losses[jit][0], f"{name}: loss did not decrease (jit={jit})")
    xla_models[name] = model
    print(f"   {name:24s} fit {losses[False][0]:.3f}->{losses[False][-1]:.3f}   "
          f"fit+XLA {losses[True][0]:.3f}->{losses[True][-1]:.3f}")


# =============================================================================
# STEP 5  get_config() -> from_config()
# =============================================================================

print("\n[STEP 5] config round trip")
for name, make_optimizer in OPTIMIZERS:
    optimizer = make_optimizer()
    config = optimizer.get_config()
    rebuilt = type(optimizer).from_config(config)
    check(rebuilt.get_config() == config, f"{name}: config round trip changed values")
print(f"   {len(OPTIMIZERS)} optimizers checked")


# =============================================================================
# STEP 6  SAVE AND LOAD WITH OPTIMIZER STATE
# =============================================================================

print("\n[STEP 6] .keras save / load with optimizer state")
for name, _ in OPTIMIZERS:
    model = xla_models[name]
    path = os.path.join(temporary_folder, f"{name}.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    check(np.array_equal(model.predict(X, verbose=0), loaded.predict(X, verbose=0), equal_nan=True),
          f"{name}: predictions differ after reload")
    check(type(loaded.optimizer) is type(model.optimizer), f"{name}: optimizer class differs after reload")
    check(loaded.optimizer.get_config() == model.optimizer.get_config(), f"{name}: optimizer config differs")
    check(len(loaded.optimizer.variables) == len(model.optimizer.variables), f"{name}: number of optimizer variables")
    same_variables = all(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
                         for a, b in zip(model.optimizer.variables, loaded.optimizer.variables))
    check(same_variables, f"{name}: optimizer variables differ after reload")
    loaded.fit(X, Y, epochs=1, batch_size=16, verbose=0)      # training continues
print(f"   {len(OPTIMIZERS)} optimizers checked")
del xla_models, model, loaded                                 # free the 27 models of STEP 4
gc.collect()


# =============================================================================
# STEP 7  LEARNING-RATE SCHEDULE + clipnorm
# =============================================================================

print("\n[STEP 7] CosineDecay schedule + clipnorm=1.0")
for name, make_optimizer in OPTIMIZERS:
    config = make_optimizer().get_config()
    other_arguments = {k: v for k, v in config.items() if k not in ("learning_rate", "clipnorm", "name")}
    schedule = keras.optimizers.schedules.CosineDecay(1e-2, decay_steps=50)
    optimizer = type(make_optimizer())(learning_rate=schedule, clipnorm=1.0, **other_arguments)
    keras.backend.clear_session()
    keras.utils.set_random_seed(1)
    initializer = keras.initializers.GlorotUniform(seed=42)
    model = keras.Sequential([
        keras.Input(shape=(4,)),
        keras.layers.Dense(16, activation="tanh", kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="tanh", kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation="softmax", kernel_initializer=initializer),
    ])
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
    history = model.fit(X, Y, epochs=3, batch_size=16, verbose=0)
    check(np.all(np.isfinite(history.history["loss"])), f"{name}: schedule + clipnorm gave a non-finite loss")
print(f"   {len(OPTIMIZERS)} optimizers checked")


# =============================================================================
# STEP 8  ACTIVATIONS IN MODELS, SAVE AND LOAD
# =============================================================================
# (a) Dense(activation=fn) for two paper activations
# (b) the FractalActivation layer with a non-default parameter (n_terms=50)
# weierstrass_mandelbrot_function_xpsin with its default 100 terms diverges
# here (NaN loss, see README); the check is then that the reloaded model gives
# the same (NaN) predictions, so NaN counts as equal to NaN.

print("\n[STEP 8] fractal activations in saved models")
for activation_fn in (faf.modified_weierstrass_function_tanh, faf.weierstrass_mandelbrot_function_xpsin):
    keras.backend.clear_session()
    keras.utils.set_random_seed(2)
    initializer = keras.initializers.GlorotUniform(seed=42)
    model = keras.Sequential([
        keras.Input(shape=(4,)),
        keras.layers.Dense(16, activation=activation_fn, kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation=activation_fn, kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation="softmax", kernel_initializer=initializer),
    ])
    model.compile(optimizer=M.AdaptiveMemoryFAdam(learning_rate=1e-2, vderiv=1.25),
                  loss="sparse_categorical_crossentropy")
    model.fit(X, Y, epochs=2, batch_size=16, verbose=0)
    path = os.path.join(temporary_folder, "activation.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    ok = check(np.array_equal(model.predict(X, verbose=0), loaded.predict(X, verbose=0), equal_nan=True),
               f"Dense(activation={activation_fn.__name__}): predictions differ after reload")
    print(f"   Dense(activation={activation_fn.__name__}): save/load {'ok' if ok else 'FAILED'}")

keras.backend.clear_session()
keras.utils.set_random_seed(3)
inputs = keras.Input(shape=(4,))
hidden = keras.layers.Dense(16)(inputs)
hidden = faf.FractalActivation("modified_weierstrass_function_tanh", n_terms=50)(hidden)
outputs = keras.layers.Dense(3, activation="softmax")(hidden)
model = keras.Model(inputs, outputs)
model.compile(optimizer=C.MemoryFAdam(learning_rate=1e-2, vderiv=0.75, history_size=6),
              loss="sparse_categorical_crossentropy")
model.fit(X, Y, epochs=2, batch_size=16, verbose=0)
# the layer must compute exactly the function with n_terms=50
reference_dense = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(16)])
reference_dense.layers[0].set_weights(model.layers[1].get_weights())
expected = faf.modified_weierstrass_function_tanh(reference_dense(X), n_terms=50)
got = keras.Model(inputs, model.layers[2].output)(X)
check(np.array_equal(np.asarray(expected), np.asarray(got), equal_nan=True),
      "FractalActivation(n_terms=50) differs from the function")
path = os.path.join(temporary_folder, "layer.keras")
model.save(path)
loaded = keras.models.load_model(path)
check(np.array_equal(model.predict(X, verbose=0), loaded.predict(X, verbose=0), equal_nan=True),
      "FractalActivation: reload")
check(loaded.layers[2].function_kwargs == {"n_terms": 50}, "FractalActivation: parameters not restored")
print("   FractalActivation layer (n_terms=50): matches the function, save/load ok")


# =============================================================================
# STEP 9  ALL ACTIVATIONS UNDER XLA
# =============================================================================

if RUN_XLA_ACTIVATION_CHECK:
    print("\n[STEP 9] every fractal activation in model.fit with jit_compile=True")
    rng = np.random.default_rng(0)
    X_small = rng.uniform(size=(64, 4)).astype(np.float32)
    Y_small = rng.integers(0, 3, 64).astype(np.int32)
    for name, activation_fn in faf.ALL_FRACTAL_ACTIVATIONS.items():
        keras.backend.clear_session()
        keras.utils.set_random_seed(0)
        model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(8, activation=activation_fn),
                                  keras.layers.Dense(3, activation="softmax")])
        model.compile(optimizer=M.AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=1.25),
                      loss="sparse_categorical_crossentropy", jit_compile=True)
        history = model.fit(X_small, Y_small, epochs=2, batch_size=16, verbose=0)
        ok = check(np.all(np.isfinite(history.history["loss"])), f"{name}: non-finite loss under XLA")
        print(f"   {name:55s} XLA fit {'ok' if ok else 'FAILED'}, loss {history.history['loss'][-1]:.4f}")


# =============================================================================
# STEP 10  MEMORY
# =============================================================================
# Six decaying-cosine models, one after another, with optimizers from all
# four families. The resident memory must not grow by the size of a training
# graph per model (see README: TensorFlow's all-reduce keeps every traced
# graph alive; our optimizers skip it when there is only one replica).

if RUN_MEMORY_CHECK and not os.path.exists("/proc/self/status"):
    print("\n[STEP 10] skipped: the memory reading uses /proc (Linux only)")
if RUN_MEMORY_CHECK and os.path.exists("/proc/self/status"):
    print("\n[STEP 10] memory over six consecutive models")
    X_memory = np.random.default_rng(0).random((105, 4)).astype(np.float32)
    Y_memory = np.random.default_rng(1).integers(0, 3, 105).astype(np.int32)
    memory_optimizers = [
        lambda: M.AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=0.75),
        lambda: C.MemoryFAdam(learning_rate=1e-3, vderiv=0.75, history_size=6),
        lambda: B.FAdam(learning_rate=1e-3, vderiv=0.75),
        lambda: D.AdaGL(learning_rate=1e-3, alpha=0.75),
        lambda: D.AOFGD_Adam(learning_rate=1e-3, alpha_init=0.75, alpha_max=1.5),
        lambda: M.AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=0.75),
    ]
    readings = []
    for i, make_optimizer in enumerate(memory_optimizers):
        keras.backend.clear_session()
        keras.utils.set_random_seed(1)
        model = keras.Sequential([
            keras.Input(shape=(4,)),
            keras.layers.Dense(32, activation=faf.decaying_cosine_function_tf),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation=faf.decaying_cosine_function_tf),
            keras.layers.Dense(3, activation="softmax"),
        ])
        model.compile(optimizer=make_optimizer(), loss="sparse_categorical_crossentropy")
        model.fit(X_memory, Y_memory, epochs=2, batch_size=16, verbose=0)
        del model
        gc.collect()
        with open("/proc/self/status") as status_file:
            rss = [int(line.split()[1]) / 1024 for line in status_file if line.startswith("VmRSS")]
        readings.append(rss[0])
        print(f"   model {i + 1}: resident memory {readings[-1]:.0f} MB")
    growth_per_model = (readings[-1] - readings[1]) / (len(readings) - 2)
    print(f"   average growth per model after the second one: {growth_per_model:.1f} MB")
    check(not growth_per_model > MEMORY_LIMIT_MB_PER_MODEL,
          f"memory grew by {growth_per_model:.1f} MB per model")


# =============================================================================
# STEP 11  SUMMARY
# =============================================================================

print(f"\nfinished in {time.time() - start_time:.0f} s")
if failures:
    print(f"{len(failures)} FAILURES:")
    for f in failures:
        print("   ", f)
    sys.exit(1)
print(f"ALL PASSED: {len(OPTIMIZERS)} optimizers and {len(faf.ALL_FRACTAL_ACTIVATIONS)} activations.")
