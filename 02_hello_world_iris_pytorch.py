#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_hello_world_iris_pytorch.py
==============================

A "hello world" for the PyTorch versions of our fractional optimizers and
fractal activation functions, on the Iris data set, with ONE single run
(one seed). It is the PyTorch twin of 01_hello_world_iris_tensorflow.py: same
steps, same settings, same network. The script is written top to bottom on
purpose: no main(), no factories, no class hierarchy. Every step is spelled
out, so you can read it like a recipe and copy the parts you need.

What happens, step by step
--------------------------
STEP  0  Settings (everything you might want to change is here)
STEP  1  Imports and versions
STEP  2  Reproducibility: one seed for Python, NumPy and PyTorch
STEP  3  Load the Iris data set
STEP  4  Train/test split (70/30) and Min-Max scaling, as in the paper
STEP  5  A first look at the fractal activation functions
STEP  6  Build the network (the paper's Iris architecture) with a fractal activation
STEP  7  Create one fractional optimizer and explain its knobs
STEP  8  Train with a plain PyTorch training loop
STEP  9  Evaluate: accuracy, macro precision / recall / F1, confusion matrix
STEP 10  Look inside the optimizer (GL coefficients, DC gain, mixing weight lambda_t)
STEP 11  Save model + optimizer (state_dict) and load them again
STEP 12  Optimizer tour: every optimizer once, same activation, same seed
STEP 13  Activation tour: every activation once, same optimizer, same seed
STEP 14  Training loop with loss feedback (set_current_loss)
STEP 15  Write all results to disk

Run it with
    python 02_hello_world_iris_pytorch.py

(no command-line arguments: every setting is in STEP 0 below)

Requirements: see README.md (conda environment). Files needed in the same
folder:
    class_b_fractional_optimizers_OHA_torch.py
    class_c_fractional_optimizers_SR_torch.py
    class_minimal_fractional_adaptive_optimizers_torch.py
    class_d_related_work_optimizers_torch.py
    fractal_activation_functions_torch.py
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
# Any name from fractal_activation_functions_torch.ALL_FRACTAL_ACTIVATIONS works.
MAIN_ACTIVATION_NAME = "decaying_cosine_function_tf"

# Which parts to run. The two tours train many small models (about 35 and 20)
# and take a few minutes on a laptop CPU; switch them off for a quick check.
RUN_OPTIMIZER_TOUR = True
RUN_ACTIVATION_TOUR = True
RUN_CUSTOM_LOOP = True
MAKE_PLOTS = True          # needs matplotlib; skipped if it is not installed

# "cuda" if a GPU is available, otherwise "cpu". Everything works on both.
DEVICE = "auto"

PRINT_EVERY_EPOCH = True   # one line per epoch in STEP 8
RESULTS_DIR = "results_02_hello_world_iris_pytorch"


# =============================================================================
# STEP 1  IMPORTS AND VERSIONS
# =============================================================================

import json
import os
import platform
import random
import time

import numpy as np
import pandas as pd
import sklearn
import torch

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

# --- our own PyTorch modules ---------------------------------------------------
# Family 1: Herrera-style fractional optimizers (gradient scaled by f_nu(w))
from class_b_fractional_optimizers_OHA_torch import (
    FSGD, FAdam, FAdagrad, FAdadelta, FRMSprop, FAdamW,
)
# Family 2: explicit Grunwald-Letnikov memory optimizers (fixed order, K past gradients)
from class_c_fractional_optimizers_SR_torch import (
    MemoryFSGD, MemoryFRMSprop, MemoryFAdam, MemoryFAdadelta,
    MemoryFAdagrad, MemoryFAdamW, MemoryFNadam, MemoryFAdamax,
)
# Family 3: adaptive memory optimizers (memory term mixed in with weight lambda_t)
from class_minimal_fractional_adaptive_optimizers_torch import (
    AdaptiveMemoryFSGD, AdaptiveMemoryFRMSprop, AdaptiveMemoryFAdam,
    AdaptiveMemoryFAdadelta, AdaptiveMemoryFAdagrad, AdaptiveMemoryFAdamW,
    AdaptiveMemoryFNadam, AdaptiveMemoryFAdamax,
)
# Family 4: published related work (AdaGL, FCSGD_GL / FCAdam_GL, AOFGD)
from class_d_related_work_optimizers_torch import (
    AdaGL, FCSGD_GL, FCAdam_GL, AOFGD_SGD, AOFGD_Adam,
)
# Fractal activation functions
import fractal_activation_functions_torch as faf

if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)

print("=" * 78)
print("Hello world: fractional optimizers x fractal activations on Iris (PyTorch)")
print("=" * 78)
print(f"Python       {platform.python_version()}")
print(f"PyTorch      {torch.__version__}")
print(f"NumPy        {np.__version__}")
print(f"scikit-learn {sklearn.__version__}")
print(f"pandas       {pd.__version__}")
print(f"device       {device}")

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# STEP 2  REPRODUCIBILITY
# =============================================================================
# We seed Python's `random`, NumPy and PyTorch. torch.manual_seed also seeds
# the GPU generators. We call it again before every model below, so every
# model starts from the same random state (same weights, same Dropout masks,
# same mini-batch order).

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
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
    y_all = label_encoder.fit_transform(iris_openml.target).astype(np.int64)
    class_names = [str(c) for c in label_encoder.classes_]
    feature_names = [str(c) for c in iris_openml.data.columns]
    data_source = "OpenML iris (version 1)"
else:
    iris_sklearn = load_iris()
    X_all = iris_sklearn.data.astype(np.float32)
    y_all = iris_sklearn.target.astype(np.int64)
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

# PyTorch tensors on the chosen device (the data set is tiny, so it simply
# stays on the device; for big data you would use a DataLoader).
X_train_t = torch.tensor(X_train, device=device)
y_train_t = torch.tensor(y_train, device=device)
X_test_t = torch.tensor(X_test, device=device)
y_test_t = torch.tensor(y_test, device=device)

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
#
# (The names keep their "_tf" suffixes so that the same names work in the
# Keras and in the PyTorch version.)

x_probe = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
probe_table = {"x": x_probe.numpy()}
for activation_name, activation_fn in faf.PAPER_FRACTAL_ACTIVATIONS.items():
    probe_table[activation_name] = activation_fn(x_probe).numpy()
probe_table["tanh (reference)"] = torch.tanh(x_probe).numpy()

print("\n[STEP 5] values of the four paper activations at a few points:")
print(pd.DataFrame(probe_table).round(4).to_string(index=False))

# Optional: plot every activation on [-3, 3].
if MAKE_PLOTS:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x_plot = torch.linspace(-3, 3, 3001)
        names = list(faf.ALL_FRACTAL_ACTIVATIONS)
        n_cols = 3
        n_rows = int(np.ceil(len(names) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.6 * n_rows))
        for ax, activation_name in zip(axes.ravel(), names):
            torch.manual_seed(SEED)  # fixes the depth of the random-depth variants
            y_plot = faf.ALL_FRACTAL_ACTIVATIONS[activation_name](x_plot).numpy()
            ax.plot(x_plot.numpy(), y_plot, lw=0.7)
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
# The paper's multi-class architecture (as in the Keras version):
#   Linear(n) -> act -> BatchNorm -> Dropout(0.2) -> Linear(2n) -> act -> BatchNorm
#   -> Dropout(0.2) -> Linear(classes)
# Glorot(Xavier)-uniform weights with a fixed seed (42) and zero biases, as
# Keras' Dense layer does. BatchNorm1d(momentum=0.01, eps=1e-3) is Keras'
# BatchNormalization(momentum=0.99, epsilon=1e-3) (PyTorch counts the momentum
# the other way round). The last layer has no softmax: CrossEntropyLoss takes
# the raw scores (logits) and applies log-softmax itself, which is the same
# loss as Keras' softmax + sparse_categorical_crossentropy.
#
# Two equivalent ways to use a fractal activation:
#   (a) call the function in your own forward(): h = faf.decaying_cosine_function_tf(h)
#   (b) put the module faf.FractalActivation("decaying_cosine_function_tf") into
#       nn.Sequential, optionally with parameters, e.g.
#       faf.FractalActivation("modified_weierstrass_function_tanh", n_terms=50)
# We use (b) here.

torch.manual_seed(SEED)
model = torch.nn.Sequential(
    torch.nn.Linear(n_features, HIDDEN_UNITS),
    faf.FractalActivation(MAIN_ACTIVATION_NAME),
    torch.nn.BatchNorm1d(HIDDEN_UNITS, momentum=0.01, eps=1e-3),
    torch.nn.Dropout(DROPOUT_RATE),
    torch.nn.Linear(HIDDEN_UNITS, 2 * HIDDEN_UNITS),
    faf.FractalActivation(MAIN_ACTIVATION_NAME),
    torch.nn.BatchNorm1d(2 * HIDDEN_UNITS, momentum=0.01, eps=1e-3),
    torch.nn.Dropout(DROPOUT_RATE),
    torch.nn.Linear(2 * HIDDEN_UNITS, n_classes),
)
for layer in model:
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight, generator=torch.Generator().manual_seed(42))
        torch.nn.init.zeros_(layer.bias)
model = model.to(device)
loss_function = torch.nn.CrossEntropyLoss()

print(f"\n[STEP 6] network with activation '{MAIN_ACTIVATION_NAME}':")
for line in str(model).splitlines():
    print("         " + line)
print(f"         trainable parameters: {sum(p.numel() for p in model.parameters())}")


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
#   lr (or learning_rate)    Adam step size
#   vderiv                   fractional order nu (fixed)
#   history_size             K, number of past gradients in the GL sum
#   normalize_coefficients   divide the GL coefficients by sum |c_k|
#   initial_mix / min_mix / max_mix   start value and bounds of lambda_t
#   adapt_step               how fast lambda_t moves per step
#   stability_threshold      gradient-change level that counts as "stable"
#   mix_ema_gamma, stability_ema_gamma   smoothing of lambda_t and of the stability score
#   warmup_steps             linear ramp of max_mix (0 = off)
# Every optimizer is a normal torch.optim.Optimizer: the first argument is
# model.parameters() (or a list of parameter groups with their own settings),
# and it works with torch.optim.lr_scheduler.*, torch.nn.utils.clip_grad_norm_,
# torch.amp, state_dict()/load_state_dict() and torch.compile'd models.

optimizer = AdaptiveMemoryFAdam(
    model.parameters(),
    lr=LR_ADAM,
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
print("         settings: " + json.dumps(optimizer.defaults, default=str))


# =============================================================================
# STEP 8  TRAIN
# =============================================================================
# The standard PyTorch loop: for every mini-batch
#   1. optimizer.zero_grad()   clear the old gradients
#   2. loss = ...              forward pass
#   3. loss.backward()         gradients
#   4. optimizer.step()        the fractional update
# model.train() switches Dropout on and lets BatchNorm use batch statistics;
# model.eval() (STEP 9) switches to inference mode.

history = {"loss": [], "accuracy": []}
batch_order_generator = torch.Generator().manual_seed(SEED)   # mini-batch shuffling

print(f"\n[STEP 8] training for {EPOCHS} epochs, batch size {BATCH_SIZE} ...")
train_start = time.perf_counter()
for epoch in range(EPOCHS):
    model.train()
    order = torch.randperm(len(X_train_t), generator=batch_order_generator).to(device)
    epoch_loss_sum, epoch_correct = 0.0, 0
    for start in range(0, len(order), BATCH_SIZE):
        batch_index = order[start:start + BATCH_SIZE]
        x_batch, y_batch = X_train_t[batch_index], y_train_t[batch_index]

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_function(logits, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss_sum += loss.item() * len(batch_index)
        epoch_correct += (logits.argmax(dim=1) == y_batch).sum().item()
    history["loss"].append(epoch_loss_sum / len(X_train_t))
    history["accuracy"].append(epoch_correct / len(X_train_t))
    if PRINT_EVERY_EPOCH:
        print(f"         epoch {epoch + 1:3d}/{EPOCHS}  loss {history['loss'][-1]:.4f}  "
              f"accuracy {history['accuracy'][-1]:.4f}")
training_time = time.perf_counter() - train_start
print(f"         training time {training_time:.2f} s")


# =============================================================================
# STEP 9  EVALUATE ON THE TEST SET
# =============================================================================
# Metrics as in the paper: accuracy and macro-averaged precision, recall and
# F1 (every class counts equally), plus the confusion matrix.

model.eval()                       # Dropout off, BatchNorm uses its running statistics
predict_start = time.perf_counter()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_loss = loss_function(test_logits, y_test_t).item()
    y_prob = torch.softmax(test_logits, dim=1).cpu().numpy()
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
    "final_training_loss": float(history["loss"][-1]),
    "final_training_accuracy": float(history["accuracy"][-1]),
}
cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))

# For comparison: accuracy on the TRAINING data in inference mode (BatchNorm
# uses its running statistics, Dropout is off), i.e. exactly how the test set
# is evaluated. A large gap between this and the per-epoch training accuracy
# of STEP 8 points to BatchNorm statistics that have not settled yet.
with torch.no_grad():
    train_accuracy_inference = (model(X_train_t).argmax(dim=1) == y_train_t).float().mean().item()
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
    ax[0].plot(history["loss"])
    ax[0].set_title("training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(history["accuracy"])
    ax[1].set_title("training accuracy")
    ax[1].set_xlabel("epoch")
    fig.suptitle(f"{type(optimizer).__name__} + {MAIN_ACTIVATION_NAME} (PyTorch)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "main_run_training_curves.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# STEP 10  LOOK INSIDE THE OPTIMIZER
# =============================================================================
# (a) The Grunwald-Letnikov coefficients c_k^(nu), c_0 = 1,
#     c_k = (1 - (nu + 1) / k) c_{k-1}. Their sum is the "DC gain": the
#     response of the memory filter to a constant gradient.
# (b) lambda_t after training, one value per parameter tensor.
# (c) For comparison: the plain memory optimizer's raw coefficients and the
#     Herrera scaling factor f_nu(w) = (|w| + eps)^(1 - nu) / Gamma(2 - nu).

print("\n[STEP 10] inside the optimizer")
print(f"         GL coefficients (normalized, K=4, nu={FRACTIONAL_ORDER}): "
      f"{np.round(optimizer.gl_coefficients, 4).tolist()}")
print(f"         DC gain (sum of coefficients): {optimizer.gl_coefficient_sum():+.4f}")

parameter_names = [name for name, _ in model.named_parameters()]
lambda_values = [float(v) for v in optimizer.current_mix_values()]
stability_values = [float(v) for v in optimizer.current_stability_values()]
print("         lambda_t and stability score per parameter tensor after training:")
for par_name, lam, stab in zip(parameter_names, lambda_values, stability_values):
    print(f"           {par_name:40s} lambda_t = {lam:.4f}   stability = {stab:.4f}")
# How to read this: lambda_t grows by adapt_step while the smoothed relative
# gradient change ("stability") stays below stability_threshold, and shrinks
# otherwise. With small mini-batches the gradients change a lot from step to
# step, so lambda_t is often pushed towards min_mix: the optimizer then falls
# back to (almost) plain Adam. That safe fallback is part of the design.
print(f"         (stability_threshold = {optimizer.defaults['stability_threshold']}; parameters whose "
      f"stability score stays above it are driven towards min_mix = {optimizer.defaults['min_mix']})")

dummy_parameter = [torch.nn.Parameter(torch.zeros(1))]
memory_example = MemoryFAdam(dummy_parameter, lr=LR_ADAM, vderiv=FRACTIONAL_ORDER, history_size=6)
print(f"         MemoryFAdam raw coefficients (K=6, nu={FRACTIONAL_ORDER}): "
      f"{np.round(memory_example.gl_coefficients, 4).tolist()}")
print(f"         their DC gain {memory_example.gl_coefficient_sum():+.4f}: for nu > 1 it is negative,"
      f" which is why the descent safeguard (descent_check=True) exists")

w_values = [0.01, 0.1, 0.5, 1.0, 2.0]
herrera_factors = FAdam._fractional_factor(torch.tensor(w_values), FRACTIONAL_ORDER, 1e-6)
print(f"         Herrera factor f_nu(w) for nu={FRACTIONAL_ORDER} at w={w_values}: "
      f"{np.round(herrera_factors.numpy().astype(np.float64), 4).tolist()}")


# =============================================================================
# STEP 11  SAVE AND RELOAD MODEL AND OPTIMIZER
# =============================================================================
# The PyTorch way: save the state_dicts of the model and of the optimizer.
# The optimizer state_dict contains every hyperparameter and the whole
# optimizer state (moments, gradient history, lambda_t, loss feedback), so
# training can be resumed exactly. torch.load(weights_only=True) is the safe
# loader (no arbitrary code); our state_dicts only contain tensors and numbers.

checkpoint_path = os.path.join(RESULTS_DIR, "iris_main_checkpoint.pt")
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)

checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
reloaded_model = torch.nn.Sequential(          # same architecture, fresh weights
    torch.nn.Linear(n_features, HIDDEN_UNITS),
    faf.FractalActivation(MAIN_ACTIVATION_NAME),
    torch.nn.BatchNorm1d(HIDDEN_UNITS, momentum=0.01, eps=1e-3),
    torch.nn.Dropout(DROPOUT_RATE),
    torch.nn.Linear(HIDDEN_UNITS, 2 * HIDDEN_UNITS),
    faf.FractalActivation(MAIN_ACTIVATION_NAME),
    torch.nn.BatchNorm1d(2 * HIDDEN_UNITS, momentum=0.01, eps=1e-3),
    torch.nn.Dropout(DROPOUT_RATE),
    torch.nn.Linear(2 * HIDDEN_UNITS, n_classes),
).to(device)
reloaded_model.load_state_dict(checkpoint["model"])
reloaded_optimizer = AdaptiveMemoryFAdam(reloaded_model.parameters())   # settings come from the checkpoint
reloaded_optimizer.load_state_dict(checkpoint["optimizer"])
reloaded_model.eval()
with torch.no_grad():
    reloaded_prob = torch.softmax(reloaded_model(X_test_t), dim=1).cpu().numpy()

print(f"\n[STEP 11] saved to {checkpoint_path} and loaded again")
print(f"          reloaded optimizer: {type(reloaded_optimizer).__name__}, vderiv "
      f"{reloaded_optimizer.param_groups[0]['vderiv']}, lambda_t restored: "
      f"{np.allclose([float(v) for v in reloaded_optimizer.current_mix_values()], lambda_values)}")
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
# The PyTorch baselines are set up like the Keras defaults used in the Keras
# script (eps 1e-7, RMSprop rho 0.9 = PyTorch alpha 0.9, Adadelta rho 0.95,
# Adagrad initial accumulator 0.1, AdamW weight decay 0.004).
# Our classes take `lr=` (PyTorch style) and also `learning_rate=` (Keras style).

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
    # --- classical PyTorch optimizers (baselines) ------------------------------
    ("baseline", "SGD", torch.optim.SGD, dict(lr=LR_SGD, momentum=SGD_MOMENTUM)),
    ("baseline", "Adam", torch.optim.Adam, dict(lr=LR_ADAM, eps=1e-7)),
    ("baseline", "RMSprop", torch.optim.RMSprop, dict(lr=LR_RMSPROP, alpha=0.9, eps=1e-7)),
    ("baseline", "Adadelta", torch.optim.Adadelta, dict(lr=LR_ADADELTA, rho=0.95, eps=1e-7)),
    ("baseline", "Adagrad", torch.optim.Adagrad, dict(lr=LR_ADAGRAD, initial_accumulator_value=0.1, eps=1e-7)),
    ("baseline", "AdamW", torch.optim.AdamW, dict(lr=LR_ADAM, weight_decay=0.004, eps=1e-7)),
    ("baseline", "NAdam", torch.optim.NAdam, dict(lr=LR_ADAM, eps=1e-7)),
    ("baseline", "Adamax", torch.optim.Adamax, dict(lr=LR_ADAM, eps=1e-7)),
    # --- Herrera-style: g -> g * (|w| + eps)^(1 - nu) / Gamma(2 - nu) ------------
    ("herrera", "FSGD", FSGD, dict(lr=LR_SGD, momentum=SGD_MOMENTUM, vderiv=nu)),
    ("herrera", "FAdam", FAdam, dict(lr=LR_ADAM, vderiv=nu)),
    ("herrera", "FRMSprop", FRMSprop, dict(lr=LR_RMSPROP, vderiv=nu)),
    ("herrera", "FAdadelta", FAdadelta, dict(lr=LR_ADADELTA, vderiv=nu)),
    ("herrera", "FAdagrad", FAdagrad, dict(lr=LR_ADAGRAD, vderiv=nu)),
    ("herrera", "FAdamW", FAdamW, dict(lr=LR_ADAM, weight_decay=0.004, vderiv=nu)),
    # --- explicit memory: g -> norm-matched GL sum of the last K gradients ------
    ("memory", "MemoryFSGD", MemoryFSGD, dict(lr=LR_SGD, momentum=SGD_MOMENTUM, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFAdam", MemoryFAdam, dict(lr=LR_ADAM, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFRMSprop", MemoryFRMSprop, dict(lr=LR_RMSPROP, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFAdadelta", MemoryFAdadelta, dict(lr=LR_ADADELTA, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFAdagrad", MemoryFAdagrad, dict(lr=LR_ADAGRAD, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFAdamW", MemoryFAdamW, dict(lr=LR_ADAM, weight_decay=0.004, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFNadam", MemoryFNadam, dict(lr=LR_ADAM, vderiv=nu, **memory_settings)),
    ("memory", "MemoryFAdamax", MemoryFAdamax, dict(lr=LR_ADAM, vderiv=nu, **memory_settings)),
    # --- adaptive memory: g -> (1 - lambda_t) g + lambda_t * memory term --------
    ("adaptive", "AdaptiveMemoryFSGD", AdaptiveMemoryFSGD, dict(lr=LR_SGD, momentum=SGD_MOMENTUM, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFAdam", AdaptiveMemoryFAdam, dict(lr=LR_ADAM, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFRMSprop", AdaptiveMemoryFRMSprop, dict(lr=LR_RMSPROP, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFAdadelta", AdaptiveMemoryFAdadelta, dict(lr=LR_ADADELTA, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFAdagrad", AdaptiveMemoryFAdagrad, dict(lr=LR_ADAGRAD, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFAdamW", AdaptiveMemoryFAdamW, dict(lr=LR_ADAM, weight_decay=0.004, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFNadam", AdaptiveMemoryFNadam, dict(lr=LR_ADAM, vderiv=nu, **adaptive_settings)),
    ("adaptive", "AdaptiveMemoryFAdamax", AdaptiveMemoryFAdamax, dict(lr=LR_ADAM, vderiv=nu, **adaptive_settings)),
    # --- related work ------------------------------------------------------------
    ("related", "AdaGL", AdaGL, dict(lr=LR_ADAM, alpha=nu, history_size=10)),
    ("related", "FCSGD_GL", FCSGD_GL, dict(lr=LR_SGD, momentum=SGD_MOMENTUM, alpha=nu, history_size=11, perturbation_prob=0.5)),
    ("related", "FCAdam_GL", FCAdam_GL, dict(lr=LR_ADAM, alpha=nu, history_size=11, perturbation_prob=0.5)),
    ("related", "AOFGD_SGD", AOFGD_SGD, dict(lr=LR_SGD, momentum=SGD_MOMENTUM, alpha_init=nu, **aofgd_settings)),
    ("related", "AOFGD_Adam", AOFGD_Adam, dict(lr=LR_ADAM, alpha_init=nu, **aofgd_settings)),
]

optimizer_tour_rows = []
if RUN_OPTIMIZER_TOUR:
    print(f"\n[STEP 12] optimizer tour: {len(OPTIMIZER_TOUR)} optimizers, "
          f"activation '{MAIN_ACTIVATION_NAME}', nu = {nu}")

    for family, opt_name, opt_class, opt_kwargs in OPTIMIZER_TOUR:
        # same seed -> every optimizer starts from the same weights and batches
        torch.manual_seed(SEED)
        tour_model = torch.nn.Sequential(
            torch.nn.Linear(n_features, HIDDEN_UNITS),
            faf.FractalActivation(MAIN_ACTIVATION_NAME),
            torch.nn.BatchNorm1d(HIDDEN_UNITS, momentum=0.01, eps=1e-3),
            torch.nn.Dropout(DROPOUT_RATE),
            torch.nn.Linear(HIDDEN_UNITS, 2 * HIDDEN_UNITS),
            faf.FractalActivation(MAIN_ACTIVATION_NAME),
            torch.nn.BatchNorm1d(2 * HIDDEN_UNITS, momentum=0.01, eps=1e-3),
            torch.nn.Dropout(DROPOUT_RATE),
            torch.nn.Linear(2 * HIDDEN_UNITS, n_classes),
        )
        for layer in tour_model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, generator=torch.Generator().manual_seed(42))
                torch.nn.init.zeros_(layer.bias)
        tour_model = tour_model.to(device)
        tour_optimizer = opt_class(tour_model.parameters(), **opt_kwargs)   # a NEW optimizer per model

        t0 = time.perf_counter()
        tour_generator = torch.Generator().manual_seed(SEED)
        for epoch in range(EPOCHS):
            tour_model.train()
            order = torch.randperm(len(X_train_t), generator=tour_generator).to(device)
            epoch_loss_sum = 0.0
            for start in range(0, len(order), BATCH_SIZE):
                batch_index = order[start:start + BATCH_SIZE]
                tour_optimizer.zero_grad()
                loss = loss_function(tour_model(X_train_t[batch_index]), y_train_t[batch_index])
                loss.backward()
                tour_optimizer.step()
                epoch_loss_sum += loss.item() * len(batch_index)
        final_train_loss = epoch_loss_sum / len(X_train_t)
        t_train = time.perf_counter() - t0

        tour_model.eval()
        with torch.no_grad():
            tour_pred = tour_model(X_test_t).argmax(dim=1).cpu().numpy()

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
            "final_train_loss": final_train_loss,
            "train_time_s": t_train,
            "diverged": bool(not np.isfinite(final_train_loss)),
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
#     is the expected behaviour of these functions, not a bug (with
#     cantor_set_activation_tf the layers before it get no gradient at all,
#     p.grad stays None, and the optimizer skips them);
#   - the Weierstrass-Mandelbrot variants and the plain Weierstrass function
#     sum terms 2^(-k/2) sin(2 pi 2^k x) for k up to 99, whose derivatives
#     grow like 2pi 2^(k/2) (about 10^15 for the last terms). With 100 terms
#     training can overflow to NaN; such runs are marked "diverged" (the
#     paper also reports the x + sin variant behaving very differently).
#     Fewer terms, e.g. FractalActivation("weierstrass_mandelbrot_function_xpsin",
#     num_terms=10), or gradient clipping (torch.nn.utils.clip_grad_norm_)
#     are ways to tame them.

ACTIVATION_TOUR = [("classical", "relu", torch.nn.ReLU), ("classical", "tanh", torch.nn.Tanh)]
for activation_name in faf.ALL_FRACTAL_ACTIVATIONS:
    group = "fractal (paper)" if activation_name in faf.PAPER_FRACTAL_ACTIVATIONS else "fractal"
    ACTIVATION_TOUR.append((group, activation_name, None))

activation_tour_rows = []
if RUN_ACTIVATION_TOUR:
    print(f"\n[STEP 13] activation tour: {len(ACTIVATION_TOUR)} activations, "
          f"optimizer AdaptiveMemoryFAdam, nu = {nu}")

    for group, activation_name, classical_module in ACTIVATION_TOUR:
        torch.manual_seed(SEED)
        if classical_module is not None:
            act_1, act_2 = classical_module(), classical_module()
        else:
            act_1, act_2 = faf.FractalActivation(activation_name), faf.FractalActivation(activation_name)
        tour_model = torch.nn.Sequential(
            torch.nn.Linear(n_features, HIDDEN_UNITS),
            act_1,
            torch.nn.BatchNorm1d(HIDDEN_UNITS, momentum=0.01, eps=1e-3),
            torch.nn.Dropout(DROPOUT_RATE),
            torch.nn.Linear(HIDDEN_UNITS, 2 * HIDDEN_UNITS),
            act_2,
            torch.nn.BatchNorm1d(2 * HIDDEN_UNITS, momentum=0.01, eps=1e-3),
            torch.nn.Dropout(DROPOUT_RATE),
            torch.nn.Linear(2 * HIDDEN_UNITS, n_classes),
        )
        for layer in tour_model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, generator=torch.Generator().manual_seed(42))
                torch.nn.init.zeros_(layer.bias)
        tour_model = tour_model.to(device)
        tour_optimizer = AdaptiveMemoryFAdam(tour_model.parameters(), lr=LR_ADAM, vderiv=nu,
                                             **adaptive_settings)

        t0 = time.perf_counter()
        tour_generator = torch.Generator().manual_seed(SEED)
        for epoch in range(EPOCHS):
            tour_model.train()
            order = torch.randperm(len(X_train_t), generator=tour_generator).to(device)
            epoch_loss_sum = 0.0
            for start in range(0, len(order), BATCH_SIZE):
                batch_index = order[start:start + BATCH_SIZE]
                tour_optimizer.zero_grad()
                loss = loss_function(tour_model(X_train_t[batch_index]), y_train_t[batch_index])
                loss.backward()
                tour_optimizer.step()
                epoch_loss_sum += loss.item() * len(batch_index)
        final_train_loss = epoch_loss_sum / len(X_train_t)
        t_train = time.perf_counter() - t0

        tour_model.eval()
        with torch.no_grad():
            tour_pred = tour_model(X_test_t).argmax(dim=1).cpu().numpy()

        row = {
            "group": group,
            "activation": activation_name,
            "accuracy": accuracy_score(y_test, tour_pred),
            "precision_macro": precision_score(y_test, tour_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, tour_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, tour_pred, average="macro", zero_division=0),
            "final_train_loss": final_train_loss,
            "train_time_s": t_train,
            "diverged": bool(not np.isfinite(final_train_loss)),
        }
        activation_tour_rows.append(row)
        print(f"   {group:16s} {activation_name:54s} acc {row['accuracy']:.4f}  "
              f"F1 {row['f1_macro']:.4f}  {t_train:5.1f} s"
              + ("   <- diverged (NaN loss)" if row["diverged"] else ""))

    activation_tour = pd.DataFrame(activation_tour_rows)
    print("\n   activation tour summary:")
    print(activation_tour.round(4).to_string(index=False))


# =============================================================================
# STEP 14  TRAINING LOOP WITH LOSS FEEDBACK
# =============================================================================
# The adaptive-memory optimizers can also be told the current loss:
# set_current_loss(loss) after optimizer.step() lowers lambda_t whenever the
# smoothed loss gets worse than the best value so far (by loss_mix_decrease
# per step). It is the PyTorch counterpart of the Keras callback
# AdaptiveLossFeedback. This loop also shows two common extras:
#   torch.nn.utils.clip_grad_norm_   gradient clipping (Keras: clipnorm=1.0)
#   torch.optim.lr_scheduler.*       learning-rate schedules
# (the max_norm and the schedule are generous here, so they rarely act).

custom_loop_rows = []
if RUN_CUSTOM_LOOP:
    print("\n[STEP 14] training loop with set_current_loss(), gradient clipping and an LR schedule")
    torch.manual_seed(SEED)
    loop_model = torch.nn.Sequential(
        torch.nn.Linear(n_features, HIDDEN_UNITS),
        faf.FractalActivation(MAIN_ACTIVATION_NAME),
        torch.nn.BatchNorm1d(HIDDEN_UNITS, momentum=0.01, eps=1e-3),
        torch.nn.Dropout(DROPOUT_RATE),
        torch.nn.Linear(HIDDEN_UNITS, 2 * HIDDEN_UNITS),
        faf.FractalActivation(MAIN_ACTIVATION_NAME),
        torch.nn.BatchNorm1d(2 * HIDDEN_UNITS, momentum=0.01, eps=1e-3),
        torch.nn.Dropout(DROPOUT_RATE),
        torch.nn.Linear(2 * HIDDEN_UNITS, n_classes),
    )
    for layer in loop_model:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight, generator=torch.Generator().manual_seed(42))
            torch.nn.init.zeros_(layer.bias)
    loop_model = loop_model.to(device)
    loop_optimizer = AdaptiveMemoryFAdam(loop_model.parameters(), lr=LR_ADAM, vderiv=nu,
                                         loss_mix_decrease=0.02, **adaptive_settings)
    # cosine decay of the learning rate to 10 % over the run (one step per epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(loop_optimizer, T_max=EPOCHS,
                                                           eta_min=0.1 * LR_ADAM)

    shuffle_rng = np.random.default_rng(SEED)
    for epoch in range(EPOCHS):
        loop_model.train()
        order = shuffle_rng.permutation(len(X_train))
        epoch_losses = []
        for start in range(0, len(order), BATCH_SIZE):
            batch_index = torch.as_tensor(order[start:start + BATCH_SIZE], device=device)
            loop_optimizer.zero_grad()
            loss_value = loss_function(loop_model(X_train_t[batch_index]), y_train_t[batch_index])
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loop_model.parameters(), max_norm=10.0)
            loop_optimizer.step()
            loop_optimizer.set_current_loss(loss_value)      # <- the loss feedback
            epoch_losses.append(loss_value.item())
        scheduler.step()
        mean_lambda = float(np.mean([float(v) for v in loop_optimizer.current_mix_values()]))
        custom_loop_rows.append({"epoch": epoch + 1, "mean_loss": float(np.mean(epoch_losses)),
                                 "mean_lambda_t": mean_lambda,
                                 "learning_rate": loop_optimizer.param_groups[0]["lr"],
                                 "loss_got_worse_flag": loop_optimizer._loss_feedback["worse"]})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   epoch {epoch + 1:3d}  mean loss {np.mean(epoch_losses):.4f}  "
                  f"mean lambda_t {mean_lambda:.4f}  lr {loop_optimizer.param_groups[0]['lr']:.2e}")

    loop_model.eval()
    with torch.no_grad():
        loop_pred = loop_model(X_test_t).argmax(dim=1).cpu().numpy()
    custom_loop_metrics = {
        "accuracy": float(accuracy_score(y_test, loop_pred)),
        "f1_macro": float(f1_score(y_test, loop_pred, average="macro", zero_division=0)),
    }
    print(f"   loss-feedback loop test accuracy {custom_loop_metrics['accuracy']:.4f}, "
          f"macro F1 {custom_loop_metrics['f1_macro']:.4f}")


# =============================================================================
# STEP 15  WRITE RESULTS TO DISK
# =============================================================================

main_metrics["confusion_matrix"] = cm.tolist()
main_metrics["class_names"] = class_names
main_metrics["epoch_history"] = {k: [float(x) for x in v] for k, v in history.items()}
main_metrics["lambda_t_per_parameter"] = dict(zip(parameter_names, lambda_values))
main_metrics["versions"] = {
    "python": platform.python_version(), "torch": torch.__version__,
    "numpy": np.__version__, "scikit-learn": sklearn.__version__, "device": str(device),
}
with open(os.path.join(RESULTS_DIR, "main_run_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(main_metrics, f, indent=2)

written = ["main_run_metrics.json", "iris_main_checkpoint.pt"]
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
