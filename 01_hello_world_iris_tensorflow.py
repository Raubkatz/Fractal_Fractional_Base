#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_hello_world_iris_tensorflow.py
=================================

A "hello world" for the Keras 3 versions of our fractional optimizers and
fractal activation functions, on the Iris data set, with ONE single run
(one seed). The script is written top to bottom on purpose: no main(),
no factories, no class hierarchy. Every step is spelled out, so you can
read it like a recipe and copy the parts you need.

What happens, step by step
--------------------------
STEP  0  Settings (everything you might want to change is here)
STEP  1  Imports and versions
STEP  2  Reproducibility: one seed for Python, NumPy, TensorFlow and Keras
STEP  3  Load the Iris data set
STEP  4  Train/test split (70/30) and Min-Max scaling, as in the paper
STEP  5  A first look at the fractal activation functions
STEP  6  Build the network (the paper's Iris architecture) with a fractal activation
STEP  7  Create one fractional optimizer and explain its knobs
STEP  8  Compile and train with model.fit
STEP  9  Evaluate: accuracy, macro precision / recall / F1, confusion matrix
STEP 10  Look inside the optimizer (GL coefficients, DC gain, mixing weight lambda_t)
STEP 11  Save the trained model to a .keras file and load it again
STEP 12  Optimizer tour: every optimizer once, same activation, same seed
STEP 13  Activation tour: every activation once, same optimizer, same seed
STEP 14  Custom training loop with loss feedback (set_current_loss)
STEP 15  Write all results to disk

Run it with
    python 01_hello_world_iris_tensorflow.py

(no command-line arguments: every setting is in STEP 0 below)

Requirements: see README.md (conda environment). Files needed in the same
folder:
    class_b_tf_fractional_optimizers_OHA_keras3.py
    class_c_tf_fractional_optimizers_SR_keras3.py
    class_minimal_fractional_adaptive_optimizers_keras3.py
    class_d_related_work_optimizers_keras3.py
    fractal_activation_functions_keras3.py
"""

# =============================================================================
# STEP 0  SETTINGS
# =============================================================================

# One seed for everything. 238974 is the first run seed of the paper protocol.
SEED = 238974

# Data: sklearn ships Iris, so no download is needed. Set to True to load it
# from OpenML exactly as the paper's experiment runner did
# (fetch_openml(name="iris", version=1)). Note: scikit-learn's copy fixes two
# data points according to Fisher's paper; OpenML has the UCI version.
USE_OPENML_IRIS = False

# Paper protocol for Iris (Table "dataset hyperparameters"):
TEST_SIZE = 0.30          # 70 % train / 30 % test, split with random_state = SEED
HIDDEN_UNITS = 32         # first hidden layer has 32 units, second has 2 * 32
DROPOUT_RATE = 0.2
BATCH_SIZE = 16
EPOCHS = 30

# The fractional order nu used everywhere in this script. The paper sweeps
# nu in {0.75, 1.25, 1.5} (plus 1.75 in additional runs); nu = 1 switches
# every fractional optimizer back to its classical base optimizer.
# 0.75 is the order of the main example below (see MAIN_ACTIVATION_NAME).
FRACTIONAL_ORDER = 0.75

# Learning rates as in the paper's runner (per base-optimizer type).
LR_SGD = 1e-2
LR_ADAM = 1e-3
LR_RMSPROP = 1e-3
LR_ADADELTA = 1.0
LR_ADAGRAD = 1e-2
SGD_MOMENTUM = 0.9

# The main example (steps 6 to 11) is AdaptiveMemoryFAdam + decaying cosine
# with nu = 0.75: rank 2 of the paper's Iris table (mean accuracy 0.937,
# std 0.037 over 40 runs), and the fastest of the top three configurations.
# Any name from fractal_activation_functions_keras3.ALL_FRACTAL_ACTIVATIONS works.
MAIN_ACTIVATION_NAME = "decaying_cosine_function_tf"

# Which parts to run. The two tours train many small models (about 35 and 20)
# and take a few minutes on a laptop CPU; switch them off for a quick check.
RUN_OPTIMIZER_TOUR = True
RUN_ACTIVATION_TOUR = True
RUN_CUSTOM_LOOP = True
MAKE_PLOTS = True          # needs matplotlib; skipped if it is not installed

# XLA compilation for model.fit. "auto" (the Keras default) uses XLA on GPUs
# and plain TensorFlow graphs on CPUs. All optimizers and activations also
# work with jit_compile=True.
JIT_COMPILE = "auto"

VERBOSE_FIT = 2            # 0 = silent, 1 = progress bar, 2 = one line per epoch
RESULTS_DIR = "results_01_hello_world_iris_tensorflow"


# =============================================================================
# STEP 1  IMPORTS AND VERSIONS
# =============================================================================

import json
import os
import platform
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # less TensorFlow start-up noise
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # our classes use the TF backend

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import keras

from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --- our own Keras 3 modules -------------------------------------------------
# Family 1: Herrera-style fractional optimizers (gradient scaled by f_nu(w))
from class_b_tf_fractional_optimizers_OHA_keras3 import (
    FSGD, FAdam, FAdagrad, FAdadelta, FRMSprop, FAdamW,
)
# Family 2: explicit Grunwald-Letnikov memory optimizers (fixed order, K past gradients)
from class_c_tf_fractional_optimizers_SR_keras3 import (
    MemoryFSGD, MemoryFRMSprop, MemoryFAdam, MemoryFAdadelta,
    MemoryFAdagrad, MemoryFAdamW, MemoryFNadam, MemoryFAdamax,
)
# Family 3: adaptive memory optimizers (memory term mixed in with weight lambda_t)
from class_minimal_fractional_adaptive_optimizers_keras3 import (
    AdaptiveMemoryFSGD, AdaptiveMemoryFRMSprop, AdaptiveMemoryFAdam,
    AdaptiveMemoryFAdadelta, AdaptiveMemoryFAdagrad, AdaptiveMemoryFAdamW,
    AdaptiveMemoryFNadam, AdaptiveMemoryFAdamax,
)
# Family 4: published related work (AdaGL, FCSGD_GL / FCAdam_GL, AOFGD)
from class_d_related_work_optimizers_keras3 import (
    AdaGL, FCSGD_GL, FCAdam_GL, AOFGD_SGD, AOFGD_Adam,
)
# Fractal activation functions
import fractal_activation_functions_keras3 as faf

print("=" * 78)
print("Hello world: fractional optimizers x fractal activations on Iris (Keras 3)")
print("=" * 78)
print(f"Python       {platform.python_version()}")
print(f"TensorFlow   {tf.__version__}")
print(f"Keras        {keras.__version__}  (backend: {keras.backend.backend()})")
print(f"NumPy        {np.__version__}")
print(f"scikit-learn {sklearn.__version__}")
print(f"pandas       {pd.__version__}")
print(f"GPUs         {len(tf.config.list_physical_devices('GPU'))}")

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# STEP 2  REPRODUCIBILITY
# =============================================================================
# keras.utils.set_random_seed seeds Python's `random`, NumPy and TensorFlow
# (and Keras' own seed generator) in one call. We call it again before every
# model below, so every model starts from the same random state.

keras.utils.set_random_seed(SEED)
print(f"\n[STEP 2] seed = {SEED}")


# =============================================================================
# STEP 3  LOAD THE IRIS DATA SET
# =============================================================================
# 150 flowers, 4 numeric features (sepal length/width, petal length/width),
# 3 classes (setosa, versicolor, virginica), 50 flowers per class.

if USE_OPENML_IRIS:
    iris_openml = fetch_openml(name="iris", version=1, as_frame=True, parser="auto")
    X_all = iris_openml.data.to_numpy(dtype=np.float32)
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(iris_openml.target).astype(np.int32)
    class_names = [str(c) for c in label_encoder.classes_]
    feature_names = [str(c) for c in iris_openml.data.columns]
    data_source = "OpenML iris (version 1)"
else:
    iris_sklearn = load_iris()
    X_all = iris_sklearn.data.astype(np.float32)
    y_all = iris_sklearn.target.astype(np.int32)
    class_names = [str(c) for c in iris_sklearn.target_names]
    feature_names = [str(c) for c in iris_sklearn.feature_names]
    data_source = "scikit-learn load_iris()"

n_features = X_all.shape[1]
n_classes = len(np.unique(y_all))

print(f"\n[STEP 3] data source: {data_source}")
print(f"         X shape {X_all.shape}, classes {class_names}")
print(f"         features {feature_names}")
print(f"         samples per class {np.bincount(y_all).tolist()}")


# =============================================================================
# STEP 4  TRAIN/TEST SPLIT AND SCALING
# =============================================================================
# Same protocol as the paper: one shuffled 70/30 split with random_state =
# seed, then Min-Max scaling to [0, 1] fitted on the training part only.
# The test part is only used once, at the end, for the metrics.

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=SEED, shuffle=True
)
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

print(f"\n[STEP 4] train {X_train.shape[0]} samples, test {X_test.shape[0]} samples")
print(f"         train class counts {np.bincount(y_train, minlength=n_classes).tolist()}, "
      f"test class counts {np.bincount(y_test, minlength=n_classes).tolist()}")


# =============================================================================
# STEP 5  A FIRST LOOK AT THE FRACTAL ACTIVATION FUNCTIONS
# =============================================================================
# Every fractal activation is an ordinary function tensor -> tensor, so it
# can be called directly. faf.PAPER_FRACTAL_ACTIVATIONS holds the four used
# in the article; faf.ALL_FRACTAL_ACTIVATIONS holds all of them.
#
#   modulated_blancmange_curve            Takagi/Blancmange-type sum, tanh-modulated
#   decaying_cosine_function_tf           Weierstrass-type cosine sum, damped by exp(-|x|/2)
#   modified_weierstrass_function_tanh    Weierstrass sum * exp(-0.75|x|) + tanh(x)
#   weierstrass_mandelbrot_function_xpsin Weierstrass-Mandelbrot-type sum of x + sin(...)

x_probe = tf.constant([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=tf.float32)
probe_table = {"x": x_probe.numpy()}
for activation_name, activation_fn in faf.PAPER_FRACTAL_ACTIVATIONS.items():
    probe_table[activation_name] = activation_fn(x_probe).numpy()
probe_table["tanh (reference)"] = tf.tanh(x_probe).numpy()

print("\n[STEP 5] values of the four paper activations at a few points:")
print(pd.DataFrame(probe_table).round(4).to_string(index=False))

# Optional: plot every activation on [-3, 3].
if MAKE_PLOTS:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x_plot = np.linspace(-3, 3, 3001).astype(np.float32)
        names = list(faf.ALL_FRACTAL_ACTIVATIONS)
        n_cols = 3
        n_rows = int(np.ceil(len(names) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.6 * n_rows))
        for ax, activation_name in zip(axes.ravel(), names):
            tf.random.set_seed(SEED)  # fixes the depth of the random-depth variants
            y_plot = faf.ALL_FRACTAL_ACTIVATIONS[activation_name](tf.constant(x_plot)).numpy()
            ax.plot(x_plot, y_plot, lw=0.7)
            paper_mark = "  [paper]" if activation_name in faf.PAPER_FRACTAL_ACTIVATIONS else ""
            ax.set_title(activation_name + paper_mark, fontsize=8)
            ax.axhline(0, color="0.8", lw=0.5)
            ax.axvline(0, color="0.8", lw=0.5)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "fractal_activations.png"), dpi=150)
        plt.close(fig)
        print(f"         plot of all {len(names)} activations -> {RESULTS_DIR}/fractal_activations.png")
    except ImportError:
        MAKE_PLOTS = False
        print("         (matplotlib not installed, plots skipped)")


# =============================================================================
# STEP 6  BUILD THE NETWORK WITH A FRACTAL ACTIVATION
# =============================================================================
# The paper's multi-class architecture:
#   Dense(n) -> BatchNorm -> Dropout(0.2) -> Dense(2n) -> BatchNorm -> Dropout(0.2)
#   -> Dense(classes, softmax), Glorot-uniform weights with a fixed seed (42).
#
# Two equivalent ways to use a fractal activation:
#   (a) keras.layers.Dense(32, activation=faf.modified_weierstrass_function_tanh)
#       -> uses the default parameters (here 100 series terms)
#   (b) keras.layers.Dense(32) followed by
#       faf.FractalActivation("modified_weierstrass_function_tanh", n_terms=50)
#       -> lets you change the parameters and the model can still be saved
# We use (a) here, as in the paper.

main_activation = faf.get_fractal_activation(MAIN_ACTIVATION_NAME)

keras.backend.clear_session()
keras.utils.set_random_seed(SEED)
initializer = keras.initializers.GlorotUniform(seed=42)
model = keras.Sequential(
    [
        keras.Input(shape=(n_features,)),
        keras.layers.Dense(HIDDEN_UNITS, activation=main_activation, kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(2 * HIDDEN_UNITS, activation=main_activation, kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(n_classes, activation="softmax", kernel_initializer=initializer),
    ],
    name="iris_mlp",
)

print(f"\n[STEP 6] network with activation '{MAIN_ACTIVATION_NAME}':")
model.summary(print_fn=lambda line, *args, **kwargs: print("         " + line))


# =============================================================================
# STEP 7  CREATE ONE FRACTIONAL OPTIMIZER
# =============================================================================
# AdaptiveMemoryFAdam = Adam whose first moment is fed with
#     g_eff = (1 - lambda_t) * g_t  +  lambda_t * g_hat_t^(nu),
# where g_hat_t^(nu) is the Grunwald-Letnikov fractional gradient built from
# the last K raw gradients (norm-matched to g_t), and lambda_t is adapted
# during training (it grows while the gradients are stable, shrinks when they
# fluctuate). The second moment uses the raw gradient.
#
# Knobs (values = the paper's settings):
#   learning_rate            Adam step size
#   vderiv                   fractional order nu (fixed)
#   history_size             K, number of past gradients in the GL sum
#   normalize_coefficients   divide the GL coefficients by sum |c_k|
#   initial_mix / min_mix / max_mix   start value and bounds of lambda_t
#   adapt_step               how fast lambda_t moves per step
#   stability_threshold      gradient-change level that counts as "stable"
#   mix_ema_gamma, stability_ema_gamma   smoothing of lambda_t and of the stability score
#   warmup_steps             linear ramp of max_mix (0 = off)
# Every optimizer also accepts the Keras 3 base arguments, e.g.
#   clipnorm=1.0, global_clipnorm=..., use_ema=True, gradient_accumulation_steps=...,
#   and a learning-rate schedule instead of a float.

optimizer = AdaptiveMemoryFAdam(
    learning_rate=LR_ADAM,
    vderiv=FRACTIONAL_ORDER,
    history_size=4,
    normalize_coefficients=True,
    initial_mix=0.10,
    min_mix=0.0,
    max_mix=0.30,
    adapt_step=0.01,
    mix_ema_gamma=0.90,
    stability_ema_gamma=0.95,
    stability_threshold=0.25,
    warmup_steps=0,
)
print(f"\n[STEP 7] optimizer: {type(optimizer).__name__}")
print("         config: " + json.dumps(
    {k: v for k, v in optimizer.get_config().items() if v is not None}, default=str))


# =============================================================================
# STEP 8  COMPILE AND TRAIN
# =============================================================================

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",   # integer labels 0, 1, 2
    metrics=["accuracy"],
    jit_compile=JIT_COMPILE,
)

print(f"\n[STEP 8] training for {EPOCHS} epochs, batch size {BATCH_SIZE} ...")
train_start = time.perf_counter()
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=VERBOSE_FIT,
)
training_time = time.perf_counter() - train_start
print(f"         training time {training_time:.2f} s "
      f"(includes the one-off graph tracing of the first epoch)")


# =============================================================================
# STEP 9  EVALUATE ON THE TEST SET
# =============================================================================
# Metrics as in the paper: accuracy and macro-averaged precision, recall and
# F1 (every class counts equally), plus the confusion matrix.

test_loss, test_accuracy_keras = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)

predict_start = time.perf_counter()
y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
prediction_time = time.perf_counter() - predict_start
y_pred = np.argmax(y_prob, axis=1)

main_metrics = {
    "optimizer": type(optimizer).__name__,
    "activation": MAIN_ACTIVATION_NAME,
    "fractional_order": FRACTIONAL_ORDER,
    "seed": SEED,
    "test_loss": float(test_loss),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
    "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    "training_time_seconds": float(training_time),
    "prediction_time_seconds": float(prediction_time),
    "final_training_loss": float(history.history["loss"][-1]),
    "final_training_accuracy": float(history.history["accuracy"][-1]),
}
cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))

# For comparison: accuracy on the TRAINING data in inference mode (BatchNorm
# uses its moving statistics, Dropout is off), i.e. exactly how the test set
# is evaluated. A large gap between this and the per-epoch training accuracy
# printed by fit() points to BatchNorm statistics that have not settled yet.
_, train_accuracy_inference = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
main_metrics["train_accuracy_inference_mode"] = float(train_accuracy_inference)

print("\n[STEP 9] test-set results")
for key in ("test_loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"):
    print(f"         {key:16s} {main_metrics[key]:.4f}")
print(f"         (training accuracy, inference mode: {train_accuracy_inference:.4f}; "
      f"one seed only, the paper's run-to-run std on Iris is about 0.03-0.06)")
print("\n         confusion matrix (rows = true class, columns = predicted class):")
print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())
print("\n         classification report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))

if MAKE_PLOTS:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history.history["loss"])
    ax[0].set_title("training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(history.history["accuracy"])
    ax[1].set_title("training accuracy")
    ax[1].set_xlabel("epoch")
    fig.suptitle(f"{type(optimizer).__name__} + {MAIN_ACTIVATION_NAME}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "main_run_training_curves.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# STEP 10  LOOK INSIDE THE OPTIMIZER
# =============================================================================
# (a) The Grunwald-Letnikov coefficients c_k^(nu), c_0 = 1,
#     c_k = (1 - (nu + 1) / k) c_{k-1}. Their sum is the "DC gain": the
#     response of the memory filter to a constant gradient.
# (b) lambda_t after training, one value per trainable variable.
# (c) For comparison: the plain memory optimizer's raw coefficients and the
#     Herrera scaling factor f_nu(w) = (|w| + eps)^(1 - nu) / Gamma(2 - nu).

print("\n[STEP 10] inside the optimizer")
print(f"         GL coefficients (normalized, K=4, nu={FRACTIONAL_ORDER}): "
      f"{np.round(optimizer.gl_coefficients, 4).tolist()}")
print(f"         DC gain (sum of coefficients): {optimizer.gl_coefficient_sum():+.4f}")

variable_names = [v.path for v in model.trainable_variables]
lambda_values = [float(v) for v in optimizer.current_mix_values()]
stability_values = [float(v) for v in optimizer.current_stability_values()]
print("         lambda_t and stability score per trainable variable after training:")
for var_name, lam, stab in zip(variable_names, lambda_values, stability_values):
    print(f"           {var_name:40s} lambda_t = {lam:.4f}   stability = {stab:.4f}")
# How to read this: lambda_t grows by adapt_step while the smoothed relative
# gradient change ("stability") stays below stability_threshold, and shrinks
# otherwise. With small mini-batches the gradients change a lot from step to
# step, so lambda_t is often pushed towards min_mix: the optimizer then falls
# back to (almost) plain Adam. That safe fallback is part of the design.
print(f"         (stability_threshold = {optimizer.stability_threshold}; variables whose stability "
      f"score stays above it are driven towards min_mix = {optimizer.min_mix})")

memory_example = MemoryFAdam(learning_rate=LR_ADAM, vderiv=FRACTIONAL_ORDER, history_size=6)
print(f"         MemoryFAdam raw coefficients (K=6, nu={FRACTIONAL_ORDER}): "
      f"{np.round(memory_example.gl_coefficients, 4).tolist()}")
print(f"         their DC gain {memory_example.gl_coefficient_sum():+.4f}: for nu > 1 it is negative,"
      f" which is why the descent safeguard (descent_check=True) exists")

herrera_example = FAdam(learning_rate=LR_ADAM, vderiv=FRACTIONAL_ORDER)
w_values = [0.01, 0.1, 0.5, 1.0, 2.0]
herrera_factors = herrera_example._fractional_factor(tf.constant(w_values)).numpy().astype(np.float64)
print(f"         Herrera factor f_nu(w) for nu={FRACTIONAL_ORDER} at w={w_values}: "
      f"{np.round(herrera_factors, 4).tolist()}")


# =============================================================================
# STEP 11  SAVE AND RELOAD THE MODEL
# =============================================================================
# All optimizers and activations are registered with Keras, so the standard
# .keras format stores the architecture, the weights, the optimizer config
# and the optimizer state (moments, gradient history, lambda_t). Loading only
# needs our modules to be imported (done in STEP 1).

model_path = os.path.join(RESULTS_DIR, "iris_main_model.keras")
model.save(model_path)
reloaded_model = keras.models.load_model(model_path)
reloaded_prob = reloaded_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

print(f"\n[STEP 11] saved to {model_path} and loaded again")
print(f"          reloaded optimizer: {type(reloaded_model.optimizer).__name__}, "
      f"lambda_t restored: {np.allclose([float(v) for v in reloaded_model.optimizer.current_mix_values()], lambda_values)}")
print(f"          predictions identical after reload: {np.array_equal(y_prob, reloaded_prob)}")


# =============================================================================
# STEP 12  OPTIMIZER TOUR
# =============================================================================
# Every optimizer once, with the same network, the same activation, the same
# seed and the same data. This is a single run per optimizer, so the numbers
# show that everything works; they are not a statistical comparison (the
# paper uses 40 runs per configuration for that).
#
# Each line: (family, name, optimizer class, keyword arguments).
# For the Keras baselines we use keras.optimizers.* with `learning_rate=`
# (Keras 3 no longer accepts `lr=`; our classes still accept both).

memory_settings = dict(history_size=6, normalize_coefficients=False)
adaptive_settings = dict(
    history_size=4, normalize_coefficients=True, initial_mix=0.10, min_mix=0.0,
    max_mix=0.30, adapt_step=0.01, mix_ema_gamma=0.90, stability_ema_gamma=0.95,
    stability_threshold=0.25,
)
aofgd_settings = dict(
    alpha_min=0.5, alpha_max=max(1.5, FRACTIONAL_ORDER), delta_alpha=0.02,
    cef_threshold=0.05, ema_gamma=0.95, frac_epsilon=1e-6, eps_cef=1e-8,
)
nu = FRACTIONAL_ORDER

OPTIMIZER_TOUR = [
    # --- classical Keras optimizers (baselines) --------------------------------
    ("baseline", "SGD", keras.optimizers.SGD, dict(learning_rate=LR_SGD, momentum=SGD_MOMENTUM)),
    ("baseline", "Adam", keras.optimizers.Adam, dict(learning_rate=LR_ADAM)),
    ("baseline", "RMSprop", keras.optimizers.RMSprop, dict(learning_rate=LR_RMSPROP)),
    ("baseline", "Adadelta", keras.optimizers.Adadelta, dict(learning_rate=LR_ADADELTA)),
    ("baseline", "Adagrad", keras.optimizers.Adagrad, dict(learning_rate=LR_ADAGRAD)),
    ("baseline", "AdamW", keras.optimizers.AdamW, dict(learning_rate=LR_ADAM, weight_decay=0.004)),
    ("baseline", "Nadam", keras.optimizers.Nadam, dict(learning_rate=LR_ADAM)),
    ("baseline", "Adamax", keras.optimizers.Adamax, dict(learning_rate=LR_ADAM)),
    # --- Herrera-style: g -> g * (|w| + eps)^(1 - nu) / Gamma(2 - nu) ------------
    ("herrera", "FSGD", FSGD, dict(learning_rate=LR_SGD, momentum=SGD_MOMENTUM, vderiv=nu)),
    ("herrera", "FAdam", FAdam, dict(learning_rate=LR_ADAM, vderiv=nu)),
    ("herrera", "FRMSprop", FRMSprop, dict(learning_rate=LR_RMSPROP, vderiv=nu)),
    ("herrera", "FAdadelta", FAdadelta, dict(learning_rate=LR_ADADELTA, vderiv=nu)),
    ("herrera", "FAdagrad", FAdagrad, dict(learning_rate=LR_ADAGRAD, vderiv=nu)),
    ("herrera", "FAdamW", FAdamW, dict(learning_rate=LR_ADAM, weight_decay=0.004, vderiv=nu)),
    # --- explicit memory: g -> norm-matched GL sum of the last K gradients ------
    ("memory", "MemoryFSGD", MemoryFSGD, dict(learning_rate=LR_SGD, momentum=SGD_MOMENTUM, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFAdam", MemoryFAdam, dict(learning_rate=LR_ADAM, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFRMSprop", MemoryFRMSprop, dict(learning_rate=LR_RMSPROP, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFAdadelta", MemoryFAdadelta, dict(learning_rate=LR_ADADELTA, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFAdagrad", MemoryFAdagrad, dict(learning_rate=LR_ADAGRAD, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFAdamW", MemoryFAdamW, dict(learning_rate=LR_ADAM, weight_decay=0.004, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFNadam", MemoryFNadam, dict(learning_rate=LR_ADAM, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFAdamax", MemoryFAdamax, dict(learning_rate=LR_ADAM, vderiv=nu, **memory_settings)),
    # --- adaptive memory: g -> (1 - lambda_t) g + lambda_t * memory term --------
    ("adaptive", "AdaptiveMemoryFSGD", AdaptiveMemoryFSGD, dict(learning_rate=LR_SGD, momentum=SGD_MOMENTUM, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFAdam", AdaptiveMemoryFAdam, dict(learning_rate=LR_ADAM, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFRMSprop", AdaptiveMemoryFRMSprop, dict(learning_rate=LR_RMSPROP, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFAdadelta", AdaptiveMemoryFAdadelta, dict(learning_rate=LR_ADADELTA, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFAdagrad", AdaptiveMemoryFAdagrad, dict(learning_rate=LR_ADAGRAD, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFAdamW", AdaptiveMemoryFAdamW, dict(learning_rate=LR_ADAM, weight_decay=0.004, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFNadam", AdaptiveMemoryFNadam, dict(learning_rate=LR_ADAM, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFAdamax", AdaptiveMemoryFAdamax, dict(learning_rate=LR_ADAM, vderiv=nu, **adaptive_settings)),
    # --- related work ------------------------------------------------------------
    ("related", "AdaGL", AdaGL, dict(learning_rate=LR_ADAM, alpha=nu, history_size=10)),
    ("related", "FCSGD_GL", FCSGD_GL, dict(learning_rate=LR_SGD, momentum=SGD_MOMENTUM, alpha=nu, history_size=11, perturbation_prob=0.5)),
    ("related", "FCAdam_GL", FCAdam_GL, dict(learning_rate=LR_ADAM, alpha=nu, history_size=11, perturbation_prob=0.5)),
    ("related", "AOFGD_SGD", AOFGD_SGD, dict(learning_rate=LR_SGD, momentum=SGD_MOMENTUM, alpha_init=nu, **aofgd_settings)),
    ("related", "AOFGD_Adam", AOFGD_Adam, dict(learning_rate=LR_ADAM, alpha_init=nu, **aofgd_settings)),
]

optimizer_tour_rows = []
if RUN_OPTIMIZER_TOUR:
    print(f"\n[STEP 12] optimizer tour: {len(OPTIMIZER_TOUR)} optimizers, "
          f"activation '{MAIN_ACTIVATION_NAME}', nu = {nu}")
    tour_activation = faf.get_fractal_activation(MAIN_ACTIVATION_NAME)

    for family, opt_name, opt_class, opt_kwargs in OPTIMIZER_TOUR:
        # fresh session and seed -> every optimizer starts from the same state
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)
        initializer = keras.initializers.GlorotUniform(seed=42)
        tour_model = keras.Sequential(
            [
                keras.Input(shape=(n_features,)),
                keras.layers.Dense(HIDDEN_UNITS, activation=tour_activation, kernel_initializer=initializer),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(DROPOUT_RATE),
                keras.layers.Dense(2 * HIDDEN_UNITS, activation=tour_activation, kernel_initializer=initializer),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(DROPOUT_RATE),
                keras.layers.Dense(n_classes, activation="softmax", kernel_initializer=initializer),
            ]
        )
        tour_optimizer = opt_class(**opt_kwargs)   # a NEW optimizer for every model
        tour_model.compile(optimizer=tour_optimizer, loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"], jit_compile=JIT_COMPILE)

        t0 = time.perf_counter()
        tour_history = tour_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      shuffle=True, verbose=0)
        t_train = time.perf_counter() - t0
        tour_pred = np.argmax(tour_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)

        # optional diagnostics, only some optimizers have them
        mean_lambda = (float(np.mean([float(v) for v in tour_optimizer.current_mix_values()]))
                       if hasattr(tour_optimizer, "current_mix_values") else np.nan)
        mean_alpha = (float(np.mean([float(v) for v in tour_optimizer.current_alpha_values()]))
                      if hasattr(tour_optimizer, "current_alpha_values") else np.nan)

        row = {
            "family": family,
            "optimizer": opt_name,
            "accuracy": accuracy_score(y_test, tour_pred),
            "precision_macro": precision_score(y_test, tour_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, tour_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, tour_pred, average="macro", zero_division=0),
            "final_train_loss": tour_history.history["loss"][-1],
            "train_time_s": t_train,
            "diverged": bool(not np.isfinite(tour_history.history["loss"][-1])),
            "mean_lambda_t": mean_lambda,
            "mean_alpha_t": mean_alpha,
        }
        optimizer_tour_rows.append(row)
        print(f"   {family:9s} {opt_name:24s} acc {row['accuracy']:.4f}  F1 {row['f1_macro']:.4f}  "
              f"loss {row['final_train_loss']:.4f}  {t_train:5.1f} s"
              + ("   <- diverged (NaN loss)" if row["diverged"] else ""))

    optimizer_tour = pd.DataFrame(optimizer_tour_rows)
    print("\n   optimizer tour summary:")
    print(optimizer_tour.round(4).to_string(index=False))


# =============================================================================
# STEP 13  ACTIVATION TOUR
# =============================================================================
# Every activation once (two classical references + all fractal functions),
# with the optimizer of STEP 7. Remarks:
#   - the random_depth_* functions draw a new number of series terms in every
#     forward pass (a kind of stochastic regularisation);
#   - cantor_function_tf / cantor_set_activation_tf are piecewise constant, so
#     no gradient flows through the hidden layers: expect poor accuracy, which
#     is the expected behaviour of these functions, not a bug (Keras also
#     warns that some variables get no gradient);
#   - the Weierstrass-Mandelbrot variants and the plain Weierstrass function
#     sum terms 2^(-k/2) sin(2 pi 2^k x) for k up to 99, whose derivatives
#     grow like 2pi 2^(k/2) (about 10^15 for the last terms). With 100 terms
#     training can overflow to NaN; such runs are marked "diverged" (the
#     paper also reports the x + sin variant behaving very differently).
#     Fewer terms, e.g. FractalActivation("weierstrass_mandelbrot_function_xpsin",
#     num_terms=10), or gradient clipping (clipnorm=1.0) are ways to tame them.

ACTIVATION_TOUR = [("classical", "relu", "relu"), ("classical", "tanh", "tanh")]
for activation_name, activation_fn in faf.ALL_FRACTAL_ACTIVATIONS.items():
    group = "fractal (paper)" if activation_name in faf.PAPER_FRACTAL_ACTIVATIONS else "fractal"
    ACTIVATION_TOUR.append((group, activation_name, activation_fn))

activation_tour_rows = []
if RUN_ACTIVATION_TOUR:
    print(f"\n[STEP 13] activation tour: {len(ACTIVATION_TOUR)} activations, "
          f"optimizer AdaptiveMemoryFAdam, nu = {nu}")

    for group, activation_name, activation_fn in ACTIVATION_TOUR:
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)
        initializer = keras.initializers.GlorotUniform(seed=42)
        tour_model = keras.Sequential(
            [
                keras.Input(shape=(n_features,)),
                keras.layers.Dense(HIDDEN_UNITS, activation=activation_fn, kernel_initializer=initializer),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(DROPOUT_RATE),
                keras.layers.Dense(2 * HIDDEN_UNITS, activation=activation_fn, kernel_initializer=initializer),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(DROPOUT_RATE),
                keras.layers.Dense(n_classes, activation="softmax", kernel_initializer=initializer),
            ]
        )
        tour_optimizer = AdaptiveMemoryFAdam(learning_rate=LR_ADAM, vderiv=nu, **adaptive_settings)
        tour_model.compile(optimizer=tour_optimizer, loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"], jit_compile=JIT_COMPILE)

        t0 = time.perf_counter()
        tour_history = tour_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      shuffle=True, verbose=0)
        t_train = time.perf_counter() - t0
        tour_pred = np.argmax(tour_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)

        row = {
            "group": group,
            "activation": activation_name,
            "accuracy": accuracy_score(y_test, tour_pred),
            "precision_macro": precision_score(y_test, tour_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, tour_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, tour_pred, average="macro", zero_division=0),
            "final_train_loss": tour_history.history["loss"][-1],
            "train_time_s": t_train,
            "diverged": bool(not np.isfinite(tour_history.history["loss"][-1])),
        }
        activation_tour_rows.append(row)
        print(f"   {group:16s} {activation_name:54s} acc {row['accuracy']:.4f}  "
              f"F1 {row['f1_macro']:.4f}  {t_train:5.1f} s"
              + ("   <- diverged (NaN loss)" if row["diverged"] else ""))

    activation_tour = pd.DataFrame(activation_tour_rows)
    print("\n   activation tour summary:")
    print(activation_tour.round(4).to_string(index=False))


# =============================================================================
# STEP 14  CUSTOM TRAINING LOOP WITH LOSS FEEDBACK
# =============================================================================
# model.fit is all you need for the paper setting. If you write your own loop
# (GradientTape + apply_gradients), the adaptive-memory optimizers can also
# be told the current loss: set_current_loss(loss) lowers lambda_t whenever
# the smoothed loss gets worse than the best value so far.
# (With model.fit you get the same through the callback
#  class_minimal_fractional_adaptive_optimizers_keras3.AdaptiveLossFeedback.)

custom_loop_rows = []
if RUN_CUSTOM_LOOP:
    print("\n[STEP 14] custom training loop with set_current_loss()")
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    initializer = keras.initializers.GlorotUniform(seed=42)
    loop_model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(HIDDEN_UNITS, activation=main_activation, kernel_initializer=initializer),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(2 * HIDDEN_UNITS, activation=main_activation, kernel_initializer=initializer),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(n_classes, activation="softmax", kernel_initializer=initializer),
        ]
    )
    loop_optimizer = AdaptiveMemoryFAdam(learning_rate=LR_ADAM, vderiv=nu, loss_mix_decrease=0.02,
                                         **adaptive_settings)
    loss_function = keras.losses.SparseCategoricalCrossentropy()

    # One training step, compiled into a TensorFlow graph with tf.function.
    @tf.function
    def one_training_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = loop_model(x_batch, training=True)
            loss_value = loss_function(y_batch, predictions)
        gradients = tape.gradient(loss_value, loop_model.trainable_variables)
        loop_optimizer.apply_gradients(zip(gradients, loop_model.trainable_variables))
        return loss_value

    shuffle_rng = np.random.default_rng(SEED)
    for epoch in range(EPOCHS):
        order = shuffle_rng.permutation(len(X_train))
        epoch_losses = []
        for start in range(0, len(order), BATCH_SIZE):
            batch_index = order[start:start + BATCH_SIZE]
            loss_value = one_training_step(tf.constant(X_train[batch_index]),
                                           tf.constant(y_train[batch_index]))
            loop_optimizer.set_current_loss(loss_value)     # <- the loss feedback
            epoch_losses.append(float(loss_value))
        mean_lambda = float(np.mean([float(v) for v in loop_optimizer.current_mix_values()]))
        custom_loop_rows.append({"epoch": epoch + 1, "mean_loss": float(np.mean(epoch_losses)),
                                 "mean_lambda_t": mean_lambda,
                                 "loss_got_worse_flag": loop_optimizer._loss_worse_value})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   epoch {epoch + 1:3d}  mean loss {np.mean(epoch_losses):.4f}  "
                  f"mean lambda_t {mean_lambda:.4f}")

    loop_pred = np.argmax(loop_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
    custom_loop_metrics = {
        "accuracy": float(accuracy_score(y_test, loop_pred)),
        "f1_macro": float(f1_score(y_test, loop_pred, average="macro", zero_division=0)),
    }
    print(f"   custom loop test accuracy {custom_loop_metrics['accuracy']:.4f}, "
          f"macro F1 {custom_loop_metrics['f1_macro']:.4f}")


# =============================================================================
# STEP 15  WRITE RESULTS TO DISK
# =============================================================================

main_metrics["confusion_matrix"] = cm.tolist()
main_metrics["class_names"] = class_names
main_metrics["epoch_history"] = {k: [float(x) for x in v] for k, v in history.history.items()}
main_metrics["lambda_t_per_variable"] = dict(zip(variable_names, lambda_values))
main_metrics["versions"] = {
    "python": platform.python_version(), "tensorflow": tf.__version__,
    "keras": keras.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
}
with open(os.path.join(RESULTS_DIR, "main_run_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(main_metrics, f, indent=2)

written = ["main_run_metrics.json", "iris_main_model.keras"]
if optimizer_tour_rows:
    optimizer_tour.to_csv(os.path.join(RESULTS_DIR, "optimizer_tour.csv"), index=False)
    written.append("optimizer_tour.csv")
if activation_tour_rows:
    activation_tour.to_csv(os.path.join(RESULTS_DIR, "activation_tour.csv"), index=False)
    written.append("activation_tour.csv")
if custom_loop_rows:
    pd.DataFrame(custom_loop_rows).to_csv(os.path.join(RESULTS_DIR, "custom_loop_history.csv"), index=False)
    written.append("custom_loop_history.csv")
if MAKE_PLOTS:
    written += ["fractal_activations.png", "main_run_training_curves.png"]

print(f"\n[STEP 15] results written to {RESULTS_DIR}/: {', '.join(written)}")
print("Done.")
