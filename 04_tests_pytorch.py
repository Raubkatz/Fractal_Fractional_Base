#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_tests_pytorch.py
===================

Integration test of the PyTorch files: every optimizer and every activation
inside normal PyTorch workflows. Written top to bottom, step by step.

STEP  0  Settings
STEP  1  Imports
STEP  2  A small Iris-shaped data set
STEP  3  The 27 fractional optimizers
STEP  4  Three small helpers (new model, train some epochs, all numbers of a model)
STEP  5  Training with tanh and with a fractal activation: loss finite and decreasing
STEP  6  Checkpoint: torch.save / torch.load(weights_only=True), training continues bit-identically
STEP  7  LR schedulers: the scheduled lr is the lr that is used; lr = 0 => no change; ReduceLROnPlateau
STEP  8  Determinism and step(closure)
STEP  9  float64 and bfloat16 parameters
STEP 10  Automatic mixed precision (torch.autocast + torch.amp.GradScaler)
STEP 11  Parameter groups, frozen parameters, invalid groups
STEP 12  Argument validation
STEP 13  Sparse gradients (nn.Embedding(sparse=True)) == dense gradients
STEP 14  Bernoulli masks (FCSGD_GL / FCAdam_GL) and random depth of the activations
STEP 15  Inspection helpers and the loss feedback
STEP 16  FractalActivation module, gradients at x = 0, whole-model torch.save
STEP 17  torch.compile of a model with a fractal activation
STEP 18  Memory over 20 consecutive models
STEP 19  Summary

Run it with
    python 04_tests_pytorch.py

(no command-line arguments: every setting is in STEP 0). Needs the PyTorch
files in the same folder. Takes about 2-3 minutes on a 2-core CPU.
"""

# =============================================================================
# STEP 0  SETTINGS
# =============================================================================

EPOCHS_TRAINING = 12          # STEP 5
RUN_TORCH_COMPILE = True      # STEP 17 (needs a C++ compiler for the inductor backend)
RUN_MEMORY_CHECK = True       # STEP 18
MEMORY_LIMIT_MB = 50          # STEP 18: allowed growth over 15 models


# =============================================================================
# STEP 1  IMPORTS
# =============================================================================

import gc
import io
import os
import resource
import sys
import time

import torch

import class_b_fractional_optimizers_OHA_torch as B
import class_c_fractional_optimizers_SR_torch as C
import class_minimal_fractional_adaptive_optimizers_torch as M
import class_d_related_work_optimizers_torch as D
import fractal_activation_functions_torch as faf

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
start_time = time.time()
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

data_generator = torch.Generator().manual_seed(0)
X = torch.rand(150, 4, generator=data_generator)
Y = ((X[:, 0] * 2 + X[:, 2]).long()) % 3


# =============================================================================
# STEP 3  THE 27 FRACTIONAL OPTIMIZERS
# =============================================================================
# (name, function that creates a NEW optimizer for the given parameters)

OPTIMIZERS = [
    ("FSGD", lambda p: B.FSGD(p, lr=1e-2, momentum=0.9, vderiv=0.75)),
    ("FAdam", lambda p: B.FAdam(p, lr=1e-2, vderiv=1.25)),
    ("FAdagrad", lambda p: B.FAdagrad(p, lr=5e-2, vderiv=0.75)),
    ("FAdadelta", lambda p: B.FAdadelta(p, lr=1.0, vderiv=1.25)),
    ("FRMSprop", lambda p: B.FRMSprop(p, lr=1e-2, vderiv=0.75)),
    ("FAdamW", lambda p: B.FAdamW(p, lr=1e-2, weight_decay=0.004, vderiv=0.75)),
    ("MemoryFSGD", lambda p: C.MemoryFSGD(p, lr=1e-2, momentum=0.9, vderiv=0.75, history_size=6)),
    ("MemoryFRMSprop", lambda p: C.MemoryFRMSprop(p, lr=1e-2, vderiv=1.25, history_size=6)),
    ("MemoryFAdam", lambda p: C.MemoryFAdam(p, lr=1e-2, vderiv=1.5, history_size=6)),
    ("MemoryFAdadelta", lambda p: C.MemoryFAdadelta(p, lr=1.0, vderiv=0.75, history_size=6)),
    ("MemoryFAdagrad", lambda p: C.MemoryFAdagrad(p, lr=5e-2, vderiv=0.75, history_size=6)),
    ("MemoryFAdamW", lambda p: C.MemoryFAdamW(p, lr=1e-2, weight_decay=0.004, vderiv=0.75, history_size=6)),
    ("MemoryFNadam", lambda p: C.MemoryFNadam(p, lr=1e-2, vderiv=0.75, history_size=6)),
    ("MemoryFAdamax", lambda p: C.MemoryFAdamax(p, lr=1e-2, vderiv=0.75, history_size=6)),
    ("AdaptiveMemoryFSGD", lambda p: M.AdaptiveMemoryFSGD(p, lr=1e-2, momentum=0.9, vderiv=1.25)),
    ("AdaptiveMemoryFRMSprop", lambda p: M.AdaptiveMemoryFRMSprop(p, lr=1e-2, vderiv=1.25)),
    ("AdaptiveMemoryFAdam", lambda p: M.AdaptiveMemoryFAdam(p, lr=1e-2, vderiv=1.25)),
    ("AdaptiveMemoryFAdadelta", lambda p: M.AdaptiveMemoryFAdadelta(p, lr=1.0, vderiv=0.75)),
    ("AdaptiveMemoryFAdagrad", lambda p: M.AdaptiveMemoryFAdagrad(p, lr=5e-2, vderiv=0.75)),
    ("AdaptiveMemoryFAdamW", lambda p: M.AdaptiveMemoryFAdamW(p, lr=1e-2, weight_decay=0.004, vderiv=0.75)),
    ("AdaptiveMemoryFNadam", lambda p: M.AdaptiveMemoryFNadam(p, lr=1e-2, vderiv=0.75)),
    ("AdaptiveMemoryFAdamax", lambda p: M.AdaptiveMemoryFAdamax(p, lr=1e-2, vderiv=0.75)),
    ("AdaGL", lambda p: D.AdaGL(p, lr=1e-2, alpha=0.75, history_size=10)),
    ("FCSGD_GL", lambda p: D.FCSGD_GL(p, lr=1e-2, momentum=0.9, alpha=0.75, history_size=11)),
    ("FCAdam_GL", lambda p: D.FCAdam_GL(p, lr=1e-2, alpha=0.75, history_size=11)),
    ("AOFGD_SGD", lambda p: D.AOFGD_SGD(p, lr=1e-2, momentum=0.9, alpha_init=0.75, alpha_max=1.5)),
    ("AOFGD_Adam", lambda p: D.AOFGD_Adam(p, lr=1e-2, alpha_init=1.25, alpha_max=1.5)),
]


# =============================================================================
# STEP 4  THREE SMALL HELPERS
# =============================================================================
# These three are used by almost every step below, so they are written once.

def new_model(activation="tanh", seed=0, dropout=0.2):
    """The paper's network: Linear -> act -> BatchNorm -> Dropout, twice, then Linear."""
    torch.manual_seed(seed)
    if activation == "tanh":
        act_1, act_2 = torch.nn.Tanh(), torch.nn.Tanh()
    else:
        act_1, act_2 = faf.FractalActivation(activation), faf.FractalActivation(activation)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 32), act_1, torch.nn.BatchNorm1d(32, momentum=0.01, eps=1e-3),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(32, 16), act_2, torch.nn.BatchNorm1d(16, momentum=0.01, eps=1e-3),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(16, 3),
    )


def train_epochs(model, optimizer, n_epochs, first_epoch=0):
    """Plain training loop (batch 16, fixed batch order per epoch); returns the mean loss per epoch.
    Adaptive optimizers also get the loss feedback (set_current_loss)."""
    epoch_losses = []
    for epoch in range(first_epoch, first_epoch + n_epochs):
        model.train()
        order = torch.randperm(150, generator=torch.Generator().manual_seed(1 + epoch))
        total = 0.0
        for start in range(0, 150, 16):
            batch = order[start:start + 16]
            if len(batch) < 2:                       # BatchNorm needs at least 2 samples
                continue
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(X[batch]), Y[batch])
            loss.backward()
            optimizer.step()
            if hasattr(optimizer, "set_current_loss"):
                optimizer.set_current_loss(loss)
            total += float(loss.detach()) * len(batch)
        epoch_losses.append(total / 150)
    return epoch_losses


def all_numbers(model):
    """All parameters and buffers of a model in one vector (for exact comparisons)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()] +
                     [b.detach().reshape(-1).double() for b in model.buffers()])


# =============================================================================
# STEP 5  TRAINING
# =============================================================================

print(f"[STEP 5] training, {EPOCHS_TRAINING} epochs, with tanh and with modified_weierstrass_function_tanh")
for name, make_optimizer in OPTIMIZERS:
    line = f"   {name:24s}"
    for activation in ("tanh", "modified_weierstrass_function_tanh"):
        model = new_model(activation)
        losses = train_epochs(model, make_optimizer(model.parameters()), EPOCHS_TRAINING)
        finite = all(v == v and abs(v) < 1e6 for v in losses)
        check(finite and sum(losses[-3:]) / 3 < losses[0], f"{name}: loss not decreasing with {activation}: {losses}")
        line += f"  {activation[:8]:8s} {losses[0]:.3f}->{losses[-1]:.3f}"
    print(line)


# =============================================================================
# STEP 6  CHECKPOINT: SAVE, LOAD, CONTINUE
# =============================================================================
# 3 epochs, save model + optimizer + RNG state, 3 more epochs  ->  reference.
# Fresh model and optimizer, load the checkpoint, the same 3 epochs  ->  must
# be bit-identical (moments, gradient history, lambda_t, loss feedback, masks).

print("\n[STEP 6] checkpoint round trip")
for name, make_optimizer in OPTIMIZERS:
    torch.manual_seed(5)
    model = new_model("tanh")
    optimizer = make_optimizer(model.parameters())
    train_epochs(model, optimizer, 3)
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state()}, buffer)
    train_epochs(model, optimizer, 3, first_epoch=3)
    reference = all_numbers(model)

    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    model_2 = new_model("tanh", seed=123)                 # different weights on purpose
    optimizer_2 = make_optimizer(model_2.parameters())
    model_2.load_state_dict(checkpoint["model"])
    optimizer_2.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng"])
    train_epochs(model_2, optimizer_2, 3, first_epoch=3)
    check(torch.equal(reference, all_numbers(model_2)), f"{name}: checkpoint continuation differs")
print(f"   {len(OPTIMIZERS)} optimizers checked")


# =============================================================================
# STEP 7  LEARNING-RATE SCHEDULERS
# =============================================================================
# (a) LambdaLR(0.5) gives exactly the same result as half the lr from the start
# (b) LambdaLR(0) before the first step: the parameters do not change
# (c) ReduceLROnPlateau runs and changes the lr of every group

print("\n[STEP 7] learning-rate schedulers")
for name, make_optimizer in OPTIMIZERS:
    model = new_model("tanh", dropout=0.0)
    optimizer = make_optimizer(model.parameters())
    torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.5)
    train_epochs(model, optimizer, 1)
    with_scheduler = all_numbers(model)
    model = new_model("tanh", dropout=0.0)
    optimizer = make_optimizer(model.parameters())
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * 0.5
    train_epochs(model, optimizer, 1)
    check(torch.equal(with_scheduler, all_numbers(model)), f"{name}: LambdaLR(0.5) != lr/2")

    model = new_model("tanh", dropout=0.0).eval()          # eval: BatchNorm statistics stay fixed
    optimizer = make_optimizer(model.parameters())
    torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.0)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad()
    torch.nn.functional.cross_entropy(model(X[:32]), Y[:32]).backward()
    optimizer.step()
    check(all(torch.equal(a, b) for a, b in zip(before, model.parameters())), f"{name}: lr = 0 changed parameters")

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"]
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5)
    for value in (1.0, 2.0, 3.0):
        plateau.step(value)
    check(all(abs(g["lr"] - 0.25 * g["initial_lr"]) < 1e-15 for g in optimizer.param_groups),
          f"{name}: ReduceLROnPlateau")
print(f"   {len(OPTIMIZERS)} optimizers checked")


# =============================================================================
# STEP 8  DETERMINISM AND step(closure)
# =============================================================================

print("\n[STEP 8] determinism and step(closure)")
for name, make_optimizer in OPTIMIZERS:
    runs = []
    for repetition in range(2):
        torch.manual_seed(9)
        model = new_model("tanh")
        train_epochs(model, make_optimizer(model.parameters()), 2)
        runs.append(all_numbers(model))
    check(torch.equal(runs[0], runs[1]), f"{name}: two identical runs differ")

    model = new_model("tanh", dropout=0.0)
    optimizer = make_optimizer(model.parameters())

    def closure(optimizer=optimizer, model=model):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X), Y)
        loss.backward()
        return loss
    returned = optimizer.step(closure)
    check(isinstance(returned, torch.Tensor) and returned.ndim == 0, f"{name}: step(closure) must return the loss")
print(f"   {len(OPTIMIZERS)} optimizers checked")


# =============================================================================
# STEP 9  FLOAT64 AND BFLOAT16 PARAMETERS
# =============================================================================

print("\n[STEP 9] float64 and bfloat16 parameters")
for name, make_optimizer in OPTIMIZERS:
    for dtype in (torch.float64, torch.bfloat16):
        model = new_model("tanh", dropout=0.0).to(dtype)
        optimizer = make_optimizer(model.parameters())
        for _ in range(5):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(X.to(dtype)), Y).backward()
            optimizer.step()
        parameters_ok = all(torch.isfinite(p).all() and p.dtype == dtype for p in model.parameters())
        state_ok = all(v.dtype == dtype for st in optimizer.state.values() for v in st.values()
                       if isinstance(v, torch.Tensor) and v.is_floating_point())
        check(parameters_ok and state_ok, f"{name}: {dtype} parameters/state")
print(f"   {len(OPTIMIZERS)} optimizers checked")


# =============================================================================
# STEP 10  AUTOMATIC MIXED PRECISION
# =============================================================================

print("\n[STEP 10] torch.autocast(bfloat16) + torch.amp.GradScaler")
for name, make_optimizer in OPTIMIZERS:
    model = new_model("decaying_cosine_function_tf", dropout=0.0)
    optimizer = make_optimizer(model.parameters())
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 8)
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(X), Y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    check(all(torch.isfinite(p).all() and p.dtype == torch.float32 for p in model.parameters())
          and len(optimizer.state) > 0, f"{name}: autocast / GradScaler")
print(f"   {len(OPTIMIZERS)} optimizers checked")


# =============================================================================
# STEP 11  PARAMETER GROUPS, FROZEN PARAMETERS, INVALID GROUPS
# =============================================================================
# First Linear layer: group 0 (normal settings). Second Linear layer: group 1
# with lr = 0 and its own fractional order -> must not change. Last Linear
# layer frozen (requires_grad=False) -> no optimizer state. A group with an
# invalid order must be rejected.

print("\n[STEP 11] parameter groups")
for name, make_optimizer in OPTIMIZERS:
    model = new_model("tanh", dropout=0.0)
    first_layer, second_layer, frozen_layer = model[0], model[4], model[8]
    for p in frozen_layer.parameters():
        p.requires_grad_(False)
    optimizer = make_optimizer(first_layer.parameters())
    defaults_of_group_0 = dict(optimizer.param_groups[0])
    group_1 = {"params": list(second_layer.parameters()), "lr": 0.0}
    if "vderiv" in defaults_of_group_0:
        group_1["vderiv"] = 0.9
    if "alpha" in defaults_of_group_0:
        group_1["alpha"] = 0.9
    optimizer.add_param_group(group_1)
    before_second = [p.detach().clone() for p in second_layer.parameters()]
    before_first = [p.detach().clone() for p in first_layer.parameters()]
    for _ in range(3):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(X), Y).backward()
        optimizer.step()
    check(all(torch.equal(a, b) for a, b in zip(before_second, second_layer.parameters())),
          f"{name}: lr = 0 group changed")
    check(not all(torch.equal(a, b) for a, b in zip(before_first, first_layer.parameters())),
          f"{name}: group 0 did not change")
    check(all(p not in optimizer.state for p in frozen_layer.parameters()), f"{name}: frozen parameters got state")
    check(optimizer.param_groups[1].get("vderiv", 0.9) == 0.9 and optimizer.param_groups[1].get("alpha", 0.9) == 0.9,
          f"{name}: group 1 lost its own order")
    invalid_key = "vderiv" if "vderiv" in defaults_of_group_0 else ("alpha" if "alpha" in defaults_of_group_0 else "alpha_init")
    try:
        optimizer.add_param_group({"params": list(frozen_layer.parameters()), invalid_key: 5.0})
        check(False, f"{name}: invalid group ({invalid_key}=5) accepted")
    except ValueError:
        pass
print(f"   {len(OPTIMIZERS)} optimizers checked")


# =============================================================================
# STEP 12  ARGUMENT VALIDATION
# =============================================================================

print("\n[STEP 12] invalid arguments are rejected")
some_parameters = [torch.nn.Parameter(torch.zeros(3))]
INVALID_SETTINGS = [
    (B.FSGD, dict(vderiv=2.5), ValueError), (B.FAdam, dict(vderiv=0.0), ValueError),
    (B.FAdam, dict(beta_1=1.0), ValueError), (B.FSGD, dict(momentum=-0.1), ValueError),
    (C.MemoryFAdam, dict(history_size=0), ValueError), (C.MemoryFAdam, dict(vderiv=2.0), ValueError),
    (C.MemoryFAdam, dict(histroy_size=4), TypeError),                       # typo must not pass silently
    (M.AdaptiveMemoryFAdam, dict(min_mix=0.5, max_mix=0.2), ValueError),
    (M.AdaptiveMemoryFAdam, dict(warmup_steps=-1), ValueError),
    (M.AdaptiveMemoryFAdam, dict(inital_mix=0.1), TypeError),               # typo must not pass silently
    (D.AdaGL, dict(alpha=2.0), ValueError), (D.FCSGD_GL, dict(perturbation_prob=1.5), ValueError),
    (D.AOFGD_SGD, dict(alpha_init=0.9, alpha_max=0.8), ValueError),
    (D.AOFGD_Adam, dict(alpha_max=2.0), ValueError),
]
for optimizer_class, arguments, expected_error in INVALID_SETTINGS:
    try:
        optimizer_class(some_parameters, **arguments)
        check(False, f"{optimizer_class.__name__}({arguments}) accepted")
    except expected_error:
        pass
check(B.FAdam(some_parameters, learning_rate=0.123).param_groups[0]["lr"] == 0.123, "learning_rate= alias")
print(f"   {len(INVALID_SETTINGS)} invalid settings checked, learning_rate= alias checked")


# =============================================================================
# STEP 13  SPARSE GRADIENTS
# =============================================================================
# Unique rows, so the dense and the sparse gradient sum the same numbers.

print("\n[STEP 13] nn.Embedding(sparse=True) gives the same update as dense gradients")
rows = torch.tensor([3, 1, 7, 0, 5])
target = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
for name, make_optimizer in OPTIMIZERS:
    weights = []
    for sparse in (False, True):
        torch.manual_seed(0)
        embedding = torch.nn.Embedding(10, 4, sparse=sparse)
        optimizer = make_optimizer(embedding.parameters())
        for _ in range(4):
            optimizer.zero_grad()
            ((embedding(rows) - target) ** 2).mean().backward()
            optimizer.step()
        weights.append(embedding.weight.detach().clone())
    check(torch.equal(weights[0], weights[1]), f"{name}: sparse != dense")
print(f"   {len(OPTIMIZERS)} optimizers checked")


# =============================================================================
# STEP 14  BERNOULLI MASKS AND RANDOM DEPTH
# =============================================================================
# (a) the masks of FCSGD_GL / FCAdam_GL hit their probability (read from the
#     optimizer's own torch.rand calls)
# (b) with generator= the masks are reproducible
# (c) the random depth of the activations covers its whole range and follows torch.manual_seed

print("\n[STEP 14] Bernoulli masks and random depth")
original_rand = torch.rand
for optimizer_class in (D.FCSGD_GL, D.FCAdam_GL):
    p = torch.nn.Parameter(torch.zeros(1))
    optimizer = optimizer_class([p], lr=0.0, alpha=0.9, history_size=11, perturbation_prob=0.3)
    hits, draws = 0, 0
    for _ in range(400):
        p.grad = torch.ones(1)
        captured = []
        torch.rand = lambda *a, **k: captured.append(original_rand(*a, **k)) or captured[-1]
        try:
            optimizer.step()
        finally:
            torch.rand = original_rand
        hits += int((captured[0] < 0.3).sum())
        draws += captured[0].numel()
    check(0.25 < hits / draws < 0.35, f"{optimizer_class.__name__}: mask frequency {hits / draws:.3f}")
    print(f"   {optimizer_class.__name__}: mask frequency {hits / draws:.3f} for perturbation_prob = 0.3")

    results = []
    for repetition in range(2):
        p = torch.nn.Parameter(torch.zeros(4))
        optimizer = optimizer_class([p], lr=0.01, alpha=0.9, history_size=11, perturbation_prob=0.5,
                                    generator=torch.Generator().manual_seed(42))
        for t in range(20):
            p.grad = torch.full((4,), float(t % 3) - 1.0)
            optimizer.step()
        results.append(p.detach().clone())
    check(torch.equal(results[0], results[1]), f"{optimizer_class.__name__}: generator= not reproducible")

depths = {int(faf._sample_random_depth(10, 50)) for _ in range(3000)}
check(depths == set(range(10, 51)), f"random depth range {min(depths)}..{max(depths)}")
torch.manual_seed(0)
first = faf.random_depth_modified_weierstrass_function_tanh(torch.linspace(-2, 2, 101))
torch.manual_seed(0)
second = faf.random_depth_modified_weierstrass_function_tanh(torch.linspace(-2, 2, 101))
check(torch.equal(first, second), "random depth not reproducible with torch.manual_seed")
print(f"   random depth: values {min(depths)}..{max(depths)} drawn, reproducible with torch.manual_seed")


# =============================================================================
# STEP 15  INSPECTION HELPERS AND LOSS FEEDBACK
# =============================================================================

print("\n[STEP 15] gl_coefficients, current_mix_values, current_alpha_values, loss feedback")
optimizer = C.MemoryFAdam([torch.nn.Parameter(torch.zeros(2))], vderiv=0.75, history_size=4)
coefficients = optimizer.gl_coefficients
check(len(coefficients) == 4 and coefficients[0] == 1.0 and abs(coefficients[1] + 0.75) < 1e-15,
      f"gl_coefficients {coefficients}")
check(abs(optimizer.gl_coefficient_sum() - sum(coefficients)) < 1e-15, "gl_coefficient_sum")

# loss feedback: a steadily worse loss must push lambda_t down
final_lambda = {}
for label, losses in (("no feedback", [None] * 20), ("worsening loss", [1.0 + 0.1 * t for t in range(20)])):
    p = torch.nn.Parameter(torch.ones(3))
    optimizer = M.AdaptiveMemoryFAdam([p], lr=1e-3, vderiv=0.75, initial_mix=0.2, max_mix=0.3,
                                      loss_mix_decrease=0.05, loss_worsen_tolerance=0.0)
    for t, loss_value in enumerate(losses):
        p.grad = torch.tensor([1.0, 0.5, -0.25]) * (1 + 0.01 * t)
        optimizer.step()
        if loss_value is not None:
            optimizer.set_current_loss(loss_value)
    final_lambda[label] = float(optimizer.current_mix_values()[0])
check(final_lambda["worsening loss"] < final_lambda["no feedback"], f"loss feedback: {final_lambda}")
check(len(M.AdaptiveMemoryFAdam([torch.nn.Parameter(torch.zeros(1))]).current_mix_values()) == 0,
      "current_mix_values before the first step must be empty")

p = torch.nn.Parameter(torch.ones(3))
optimizer = D.AOFGD_Adam([p], alpha_init=1.25, alpha_max=1.5)
for t in range(5):
    p.grad = torch.ones(3) * (0.5 ** t)          # shrinking gradient -> the order goes up
    optimizer.step()
alpha = float(optimizer.current_alpha_values()[0])
check(1.25 < alpha <= 1.5, f"AOFGD alpha {alpha}")
print(f"   lambda_t after 20 steps: {final_lambda['no feedback']:.4f} without feedback, "
      f"{final_lambda['worsening loss']:.4f} with a worsening loss; AOFGD alpha {alpha:.4f}")


# =============================================================================
# STEP 16  FractalActivation MODULE
# =============================================================================
# The module equals the function; gradients at x = 0 are finite; parameters
# appear in repr; unknown names are rejected; a whole model with fractal
# activations survives torch.save / torch.load and adds no state_dict entries.

print("\n[STEP 16] FractalActivation module")
x = torch.linspace(-3, 3, 601)
for name, activation_fn in faf.ALL_FRACTAL_ACTIVATIONS.items():
    torch.manual_seed(0)
    reference = activation_fn(x)
    torch.manual_seed(0)
    check(torch.equal(faf.FractalActivation(name)(x), reference), f"FractalActivation({name}) != function")
    x0 = torch.tensor([0.0, 1e-30, -1e-30, 0.5], requires_grad=True)
    y0 = activation_fn(x0)
    if y0.requires_grad:                          # cantor_set_activation_tf has no gradient path
        y0.sum().backward()
        check(torch.isfinite(x0.grad[[0, 3]]).all(), f"{name}: non-finite gradient at x = 0")
module = faf.FractalActivation("modified_weierstrass_function_tanh", n_terms=50)
check(torch.equal(module(x), faf.modified_weierstrass_function_tanh(x, n_terms=50)), "FractalActivation kwargs")
check("n_terms=50" in repr(module), "FractalActivation repr")
try:
    faf.get_fractal_activation("nope")
    check(False, "unknown activation name accepted")
except ValueError:
    pass
model = new_model("decaying_cosine_function_tf", dropout=0.0).eval()
buffer = io.BytesIO()
torch.save(model, buffer)
buffer.seek(0)
loaded = torch.load(buffer, weights_only=False)
check(torch.equal(loaded(X), model(X)), "whole-model torch.save / torch.load")
check(set(model.state_dict()) == set(new_model("tanh").state_dict()), "FractalActivation adds state_dict entries")
print(f"   {len(faf.ALL_FRACTAL_ACTIVATIONS)} activations checked")


# =============================================================================
# STEP 17  torch.compile
# =============================================================================

if RUN_TORCH_COMPILE:
    print("\n[STEP 17] torch.compile (inductor) of a model with a fractal activation")
    compile_start = time.time()
    model = new_model("weierstrass_mandelbrot_function_xpsin", dropout=0.0).eval()
    compiled = torch.compile(model)
    x_in = X[:32].clone().requires_grad_(True)
    out_eager = model(x_in)
    out_eager.sum().backward()
    grad_eager = x_in.grad.clone()
    x_in.grad = None
    out_compiled = compiled(x_in)
    out_compiled.sum().backward()
    same = torch.allclose(out_eager, out_compiled, rtol=1e-4, atol=1e-5) and \
        torch.allclose(grad_eager, x_in.grad, rtol=1e-3, atol=1e-4)
    check(same, "torch.compile result differs from eager mode")
    print(f"   forward and backward equal to eager mode: {same} ({time.time() - compile_start:.0f} s)")


# =============================================================================
# STEP 18  MEMORY
# =============================================================================

if RUN_MEMORY_CHECK:
    print("\n[STEP 18] memory over 20 consecutive models")
    peak_memory = []
    for i in range(20):
        model = new_model("modified_weierstrass_function_tanh", seed=i)
        optimizer = M.AdaptiveMemoryFAdam(model.parameters(), lr=1e-2, vderiv=0.75)
        train_epochs(model, optimizer, 1)
        del model, optimizer
        gc.collect()
        kilobytes_per_unit = 1 / 1024 if sys.platform == "darwin" else 1     # macOS reports bytes
        peak_memory.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * kilobytes_per_unit / 1024)
    growth = peak_memory[-1] - peak_memory[4]
    check(growth < MEMORY_LIMIT_MB, f"memory grew by {growth:.0f} MB over 15 models")
    print(f"   peak memory {peak_memory[0]:.0f} MB -> {peak_memory[-1]:.0f} MB (growth after warm-up {growth:.0f} MB)")


# =============================================================================
# STEP 19  SUMMARY
# =============================================================================

print(f"\nfinished in {time.time() - start_time:.0f} s")
if failures:
    print(f"{len(failures)} FAILURES:")
    for f in failures:
        print("   ", f)
    sys.exit(1)
print(f"ALL PASSED: {len(OPTIMIZERS)} optimizers and {len(faf.ALL_FRACTAL_ACTIVATIONS)} activations.")
