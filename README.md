# Fractional Optimizers × Fractal Activations — TensorFlow/Keras 3 and PyTorch

Keras 3 (TensorFlow 2.21) **and** PyTorch 2.15 implementations of the fractional optimizers and fractal activation functions used in

> S. Raubitzek, G. Goldenits, S. Schrittwieser, P. König, K. Mallinger. *Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks.* Fractal and Fractional, 2026 (accepted). [arXiv:2608.14636](https://arxiv.org/abs/2608.14636) · Experiment code: [github.com/Raubkatz/FractalAndFractional2026](https://github.com/Raubkatz/FractalAndFractional2026)

The fractal activation functions were introduced in

> S. Raubitzek, T. Kietreiber, S. Eresheim, A. Buchelt, K. Mallinger. *Fractals in neural networks: Introducing a new class of activation functions.* Communications in Nonlinear Science and Numerical Simulation **161**, 110064 (2026). [doi:10.1016/j.cnsns.2026.110064](https://doi.org/10.1016/j.cnsns.2026.110064) · Preprint: [doi:10.20944/preprints202508.0478.v1](https://doi.org/10.20944/preprints202508.0478.v1) · Code: [github.com/Raubkatz/Fractal_Activation_Functions](https://github.com/Raubkatz/Fractal_Activation_Functions)

> **This repository collects work by many authors.** Its only purpose is to provide **one platform on which all of these optimizers can be used, side by side, with Keras 3 and with PyTorch**.
>
> **The following methods are not our work, and we do not claim them:**
> - the **Herrera-style fractional optimizers** (`FSGD`, `FAdam`, `FRMSprop`, `FAdadelta`, `FAdagrad`, `FAdamW`), developed by **Oscar Herrera-Alcántara** and co-authors [[H1]](#ref-h1), [[H2]](#ref-h2);
> - the **related-work optimizers**: **AdaGL** by Chen, Zhang & Mu [[R1]](#ref-r1); **FCSGD_GL / FCAdam_GL** by Zhou, Zhao & Huang [[R2]](#ref-r2); **AOFGD** by Xiang, Zhao, Su, Zhou, Ye & Huang [[R3]](#ref-r3);
> - the **Cantor-based activations**, by Chipera, Du & Tsapara [[C1]](#ref-c1);
> - the **classical base optimizers** (SGD, momentum, Adam, RMSprop, …) [[B1]–[B10]](#base-optimizers). The baselines used here are Keras' and PyTorch's own implementations.
>
> The ideas, the mathematics and the credit belong to these authors. Our part is limited to re-implementing their published update rules in Keras 3 and in PyTorch, from the papers, and testing them side by side. These classes are **not the original authors' code**, and any error in them is ours.
>
> **If you use any of these optimizers or activations, cite the original publications** listed below, with the links given there. You may additionally cite this repository for the Keras 3 / PyTorch implementation. Only the **GL-memory** and **adaptive-memory** optimizers come from our article [[O1]](#ref-o1), and the fractal activations from our earlier article [[O2]](#ref-o2).

## Attribution and how to cite

| What | Classes (same names in both frameworks) | Original work, please cite |
|---|---|---|
| **Herrera-style fractional optimizers** (Caputo-type gradient scaling) | `FSGD`, `FAdam`, `FAdagrad`, `FAdadelta`, `FRMSprop`, `FAdamW` | **Oscar Herrera-Alcántara** and co-authors: *Appl. Sci.* 2022 [doi:10.3390/app12189264](https://doi.org/10.3390/app12189264) [[H1]](#ref-h1); *Fractal Fract.* 2023 [doi:10.3390/fractalfract7070500](https://doi.org/10.3390/fractalfract7070500) [[H2]](#ref-h2) |
| **AdaGL** | `AdaGL` | **Chen, Zhang & Mu**, *Neural Process. Lett.* 2024 [doi:10.1007/s11063-024-11571-7](https://doi.org/10.1007/s11063-024-11571-7) [[R1]](#ref-r1) |
| **FCSGD_GL / FCAdam_GL** | `FCSGD_GL`, `FCAdam_GL` | **Zhou, Zhao & Huang**, *Mathematics* 2023 [doi:10.3390/math11020316](https://doi.org/10.3390/math11020316) [[R2]](#ref-r2) |
| **AOFGD** | `AOFGD_SGD`, `AOFGD_Adam` | **Xiang, Zhao, Su, Zhou, Ye & Huang**, SSRN preprint 2025 [doi:10.2139/ssrn.5717167](https://doi.org/10.2139/ssrn.5717167) [[R3]](#ref-r3) |
| Classical base optimizers | `keras.optimizers.*`, `torch.optim.*`, and the base rules inside every class | [[B1]–[B10]](#base-optimizers) |
| Grünwald–Letnikov derivative, short-memory principle | background of `MemoryF*`, `AdaptiveMemoryF*`, `AdaGL`, `FC*_GL` | [[G1]](#ref-g1), [[G2]](#ref-g2) |
| **Adaptive memory-fractional optimizers** (our paper) | `AdaptiveMemoryFSGD`, `…FRMSprop`, `…FAdam`, `…FAdadelta`, `…FAdagrad`, `…FAdamW`, `…FNadam`, `…FAdamax` | [[O1]](#ref-o1) |
| **GL-memory optimizers** (our implementation, with norm matching and descent safeguard) | `MemoryFSGD`, `…FRMSprop`, `…FAdam`, `…FAdadelta`, `…FAdagrad`, `…FAdamW`, `…FNadam`, `…FAdamax` | [[O1]](#ref-o1); background [[G1]](#ref-g1), [[G2]](#ref-g2); see also [[R2]](#ref-r2) |
| **Fractal activation functions** | the nine fixed-depth functions | [[O2]](#ref-o2); mathematical origins [[F1]–[F5]](#fractal-functions-mathematical-origins) |
| Random-depth variants | `random_depth_*` | extensions in this code base; cite [[O2]](#ref-o2) for the underlying functions |
| **Cantor-based activations** | `cantor_function_tf`, `cantor_set_activation_tf`, `random_depth_cantor_function_tf` | **Chipera, Du & Tsapara**, [arXiv:2512.14675](https://arxiv.org/abs/2512.14675) [[C1]](#ref-c1); [[F5]](#ref-f5) |
| Software and data | — | Keras [[S1]](#ref-s1), TensorFlow [[S2]](#ref-s2), scikit-learn [[S3]](#ref-s3), PyTorch [[S4]](#ref-s4); Iris [[D1]](#ref-d1) |

**Original code.** Herrera-Alcántara's own TensorFlow-Keras implementations are available at [ia.azc.uam.mx](http://ia.azc.uam.mx/) (data availability statement of [[H1]](#ref-h1)); [[H2]](#ref-h2) describes his PyTorch implementations. Our classes are independent re-implementations from the papers. For AdaGL, FCSGD_GL/FCAdam_GL and AOFGD we found no code published by the authors. Where the papers leave details open, the choice made here is written in the class docstring (e.g. AOFGD's convergence-evaluation factor from consecutive gradient norms, 10 historical terms for FCSGD_GL/FCAdam_GL). If a result must match an original method exactly, check it against the original publication or code.

## Files

Everything is in one folder. No packages, no command-line arguments: every script has its settings at the top and runs top to bottom, step by step.

| File | Contents |
|---|---|
| `class_b_tf_fractional_optimizers_OHA_keras3.py` / `class_b_fractional_optimizers_OHA_torch.py` | Herrera-style: `FSGD`, `FAdam`, `FAdagrad`, `FAdadelta`, `FRMSprop`, `FAdamW` |
| `class_c_tf_fractional_optimizers_SR_keras3.py` / `class_c_fractional_optimizers_SR_torch.py` | GL memory: `MemoryFSGD`, `…FRMSprop`, `…FAdam`, `…FAdadelta`, `…FAdagrad`, `…FAdamW`, `…FNadam`, `…FAdamax` |
| `class_minimal_fractional_adaptive_optimizers_keras3.py` / `…_torch.py` | Adaptive memory: `AdaptiveMemoryF…` (8 classes); Keras also `AdaptiveLossFeedback` |
| `class_d_related_work_optimizers_keras3.py` / `class_d_related_work_optimizers_torch.py` | Related work: `AdaGL`, `FCSGD_GL`, `FCAdam_GL`, `AOFGD_SGD`, `AOFGD_Adam` |
| `fractal_activation_functions_keras3.py` / `fractal_activation_functions_torch.py` | 18 fractal activations, `FractalActivation` layer / module |
| `01_hello_world_iris_tensorflow.py` | Iris, one run, every optimizer and activation, full metrics (Keras 3) |
| `02_hello_world_iris_pytorch.py` | the same in PyTorch |
| `03_tests_tensorflow.py` | Keras 3 integration test (fit, XLA, save/load, schedules, memory) |
| `04_tests_pytorch.py` | PyTorch integration test (checkpoints, schedulers, parameter groups, AMP, `torch.compile`, …) |
| `requirements.txt` | the pinned versions below |

The `_keras3` files need only TensorFlow/Keras, the `_torch` files only PyTorch. Class names, arguments and defaults are the same as in the original files and in both frameworks; in PyTorch the parameters come first and the learning rate is `lr` (`learning_rate=` also works).

## Installation

```
conda create -n fractal-fractional-keras3-torch python=3.12
conda activate fractal-fractional-keras3-torch

pip install \
    numpy==2.5.4 \
    scipy==1.18.1 \
    pandas==3.1.0 \
    scikit-learn==1.9.1 \
    tensorflow==2.21.0 \
    keras==3.15.1 \
    torch==2.15.0 \
    matplotlib==3.11.2 \
    seaborn==0.13.2
```

- Linux: the default PyTorch wheel includes CUDA (several GB). CPU only: `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`. TensorFlow GPU: `"tensorflow[and-cuda]==2.21.0"`. TensorFlow 2.21 has wheels for Python 3.10–3.13 (no Intel Macs).
- **TensorFlow and PyTorch in one Python process:** import `torch` and `torch._dynamo` *before* `tensorflow`. In the other order, the first PyTorch optimizer step can abort with `LLVM ERROR: inconsistency in registered CommandLine options` (both bundle LLVM). `05_…` shows the safe order.
- This exact list was installed into a fresh Python 3.12.13 environment, and all six scripts were run in it (`01_…` with its two tours switched off); both hello worlds reproduced the numbers below exactly.

## Running the scripts

```
python 01_hello_world_iris_tensorflow.py     # ~25 min (1-2 min with the two tours switched off in STEP 0)
python 02_hello_world_iris_pytorch.py        # ~9 min
python 03_tests_tensorflow.py                # ~16 min, ~5 GB RAM
python 04_tests_pytorch.py                   # ~3 min
python 05_tests_pytorch_vs_tensorflow.py     # ~9 min
python 06_tests_tensorflow_vs_legacy.py      # ~14 min, needs tf-keras and the original files
```

(times on a 2-core CPU). The hello worlds follow the paper's Iris protocol (70/30 split, Min-Max scaling, Dense 32 → BN → Dropout 0.2 → Dense 64 → BN → Dropout 0.2 → 3 classes, 30 epochs, batch 16, seed 238974) in 15 numbered steps: data, a first look at the activations, the network, one optimizer with every argument explained, training, accuracy / macro precision / recall / F1 / confusion matrix, a look inside the optimizer (GL coefficients, λ_t, Herrera factor), save and reload, a tour of all 35 optimizers (8 built-in baselines + 27 fractional), a tour of all 20 activations, a loop with loss feedback, and the results written to `results_01_…/` or `results_02_…/`.

Main example, **AdaptiveMemoryFAdam (ν = 0.75) + decaying cosine** (rank 2 of the paper's Iris table, 0.937 ± 0.037 over 40 runs):

| | test accuracy | macro precision | macro recall | macro F1 |
|---|---:|---:|---:|---:|
| Keras 3 (`01_…`) | 0.9111 | 0.9365 | 0.9111 | 0.9136 |
| PyTorch (`02_…`) | 0.9111 | 0.9217 | 0.9137 | 0.9153 |

The two frameworks draw different initial weights, Dropout masks and batch orders from the same seed, so their single-run numbers (also in the tours) are two independent runs, not a comparison of the frameworks; whether they compute the same thing is tested by `05_…`. One seed is not a benchmark: on Iris the paper's run-to-run standard deviation is 0.03–0.06. With 100 terms the Weierstrass–Mandelbrot variants and `weierstrass_function_tf` diverge (NaN loss), and the Cantor functions pass no gradient; both are properties of the functions. In the PyTorch activation tour `relu` scores 0.33 because BatchNorm's running statistics (paper momentum, Keras 0.99 = PyTorch 0.01) have not settled after 210 steps; in a quick check, the same network with batch statistics reached 0.93.

## Using the optimizers

```python
# Keras 3
from class_minimal_fractional_adaptive_optimizers_keras3 import AdaptiveMemoryFAdam
model.compile(optimizer=AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=0.75, history_size=4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=30, batch_size=16)

# PyTorch
from class_minimal_fractional_adaptive_optimizers_torch import AdaptiveMemoryFAdam
optimizer = AdaptiveMemoryFAdam(model.parameters(), lr=1e-3, vderiv=0.75, history_size=4)
for x, y in train_loader:
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
```

| Family | Update | Main arguments (paper settings) |
|---|---|---|
| Herrera-style [[H1]](#ref-h1), [[H2]](#ref-h2) | gradient × f_ν(w) = (\|w\| + ε)^(1−ν) / Γ(2−ν), no memory | `vderiv` (ν), `frac_epsilon=1e-6` |
| GL memory [[O1]](#ref-o1) | Σ c_k g_{t−k} over the last K gradients, c_0 = 1, c_k = (1 − (ν+1)/k) c_{k−1}; rescaled to ‖g_t‖, falls back to g_t if it does not point downhill | `vderiv`, `history_size=6`, `normalize_coefficients=False`, `norm_match`, `descent_check`, `use_raw_gradient_for_second_moment` |
| Adaptive memory [[O1]](#ref-o1) | g_eff = (1 − λ_t) g_t + λ_t · (GL term); λ_t grows while gradients are stable, shrinks otherwise | `vderiv`, `history_size=4`, `normalize_coefficients=True`, `initial_mix=0.10`, `max_mix=0.30`, `adapt_step=0.01`, `stability_threshold=0.25` |
| AdaGL [[R1]](#ref-r1) | GL gradient in both Adam moments, step-size factor C_t = 1.1 − 0.5/(1 + \|g_t − g_{t−1}\|) | `alpha`, `history_size=10` |
| FCSGD_GL / FCAdam_GL [[R2]](#ref-r2) | g_t + Σ (Bernoulli mask) · w_i g_{t−i} | `alpha`, `history_size=11`, `perturbation_prob=0.5` (PyTorch: optional `generator=`) |
| AOFGD [[R3]](#ref-r3) | Herrera-type scaling with an order α_t adapted from the ratio of consecutive gradient norms | `alpha_init`, `alpha_min=0.5`, `alpha_max=max(1.5, ν)`, `delta_alpha=0.02`, `cef_threshold=0.05`, `ema_gamma=0.95` |

Paper learning rates: SGD-type 1e-2 (momentum 0.9), Adam-type and RMSprop-type 1e-3, Adadelta-type 1.0. ν = 1 (or α = 1) gives back the base optimizer in every family. Inspection: `opt.gl_coefficients`, `opt.gl_coefficient_sum()`, `opt.current_mix_values()` (λ_t), `opt.current_stability_values()`, `opt.current_alpha_values()` (AOFGD). Loss feedback for the adaptive family: `opt.set_current_loss(loss)` after each step (Keras `model.fit`: callback `AdaptiveLossFeedback()`).

**Keras 3:** `learning_rate=` (`lr=` still works), schedules, `clipnorm`/`clipvalue`/`global_clipnorm`, `use_ema`, `gradient_accumulation_steps`, `.keras` save/load with optimizer state, XLA (`jit_compile=True`), custom `GradientTape` loops, `tf.distribute`.
**PyTorch:** a normal `torch.optim.Optimizer` with Keras argument names (`beta_1`, `beta_2`, `epsilon`, `rho`, … instead of `betas`/`eps`, so configurations copy over unchanged); per-group hyperparameters, `lr_scheduler`, `clip_grad_norm_`, `torch.autocast` + `GradScaler`, `torch.compile`, float64/bfloat16, sparse gradients, `step(closure)`, `state_dict()` with `torch.load(weights_only=True)` (resuming is bit-identical to an uninterrupted run). SGD-type classes keep the Keras momentum form (v ← μv − lr·g, w ← w + v); the `…AdamW` classes apply the weight decay after the Adam step, as in the original files.

## Using the fractal activations

```python
# Keras 3
import fractal_activation_functions_keras3 as faf
keras.layers.Dense(32, activation=faf.decaying_cosine_function_tf)                  # paper setting
faf.FractalActivation("weierstrass_mandelbrot_function_xpsin", num_terms=10)        # own parameters, saveable

# PyTorch
import fractal_activation_functions_torch as faf
h = faf.decaying_cosine_function_tf(self.fc1(x))                                    # in forward()
torch.nn.Sequential(torch.nn.Linear(4, 32), faf.FractalActivation("decaying_cosine_function_tf"))

faf.PAPER_FRACTAL_ACTIVATIONS, faf.ALL_FRACTAL_ACTIVATIONS, faf.get_fractal_activation(name)   # both
```

Paper activations: `modulated_blancmange_curve`, `decaying_cosine_function_tf`, `modified_weierstrass_function_tanh`, `weierstrass_mandelbrot_function_xpsin` (names keep their `_tf` suffix in both frameworks). The Weierstrass-type functions compute in float64 and return the input dtype. With 100 terms the Weierstrass–Mandelbrot derivatives reach ~10^15; use fewer terms or gradient clipping. The Cantor functions are piecewise constant (no gradient). At x = 0 the gradient of `modulated_blancmange_curve` is 0 in both frameworks (the PyTorch file uses TensorFlow's square-root gradient rule there). The `random_depth_*` functions draw a new depth on every call; seed with `keras.utils.set_random_seed` or `torch.manual_seed`.

## Changes compared with the original files

1. Keras 3 API (`build`, `update_step`) instead of `optimizer_v2`, which no longer exists since TensorFlow 2.16; `learning_rate=` (with `lr=` alias). The update rules are unchanged, operation for operation (`06_…`: bit-identical).
2. `set_current_loss()` now also reaches compiled training steps (the flag is an optimizer variable). Without it (the paper setting) results are identical.
3. `…AdamW` classes keep the original order (Adam step, then decay); Keras' own pre-step decay is switched off for them.
4. Activations that compute in float64 return the input dtype instead of always float32 (same result for float32 models).
5. No memory growth when many models are trained in one process: TensorFlow's all-reduce keeps every traced graph alive (~130–150 MB per model with these activations); our optimizers skip it when there is only one replica. Keras' built-in optimizers are still affected.
6. Old optimizer checkpoints cannot be loaded into the Keras 3 classes. To run the original files unchanged on TensorFlow ≥ 2.16: `pip install tf-keras==2.21.0`, set `TF_USE_LEGACY_KERAS=1` before importing TensorFlow, and set `tf.keras.optimizers.Optimizer = tf.keras.optimizers.legacy.Optimizer` before importing `class_b_…` / `class_d_…` (this is what `06_…` does).

## Verification

All results from the fresh environment above (CPU, Python 3.12.13).

| Script | What is compared | Result |
|---|---|---|
| `06_tests_tensorflow_vs_legacy.py` | Keras 3 vs the original files: 167 configurations (every class and option, ν < 1, = 1, > 1, loss feedback, Bernoulli masks), 60 steps, eager / `tf.function` / `tf.function` without Grappler; 18 activations, values and gradients | **473/473 runs bit-identical** (167 eager + 153 graph + 153 without Grappler); activations **18/18 bit-identical** |
| `05_tests_pytorch_vs_tensorflow.py` | PyTorch vs Keras 3, same 167 configurations with identical inputs, open loop (recorded gradients) and closed loop (normal training), float64 and float32; 18 activations in float32 and float64 | **167/167 pass** in each of open/closed loop × float64/float32: float64 differences ≤ 5.4·10⁻¹³ (open) and ≤ 1.7·10⁻¹² (closed); float32 within the Keras version's own float32 error. Activations **36/36 pass**: float64 ≤ 7·10⁻¹⁴, float32 Weierstrass family bit-identical |
| `03_tests_tensorflow.py` | all 27 optimizers: `fit` with and without XLA, config round trip, `.keras` save/load with optimizer state, schedule + `clipnorm`; activations in saved models and under XLA; memory | **all checks pass** for all 27 optimizers and 18 activations (memory growth 3 MB per model) |
| `04_tests_pytorch.py` | all 27 optimizers: training, checkpoint resume, schedulers, determinism, closure, float64/bfloat16, AMP, parameter groups, argument validation, sparse gradients, masks, loss feedback; `FractalActivation`, `torch.compile`, memory | **all checks pass** for all 27 optimizers and 18 activations (resume after a checkpoint bit-identical; sparse = dense updates exactly; memory flat) |

Why PyTorch vs Keras is not bit-identical: the two math libraries differ by 1–4 units in the last place in `pow`, `exp`, `sqrt`, `lgamma`. In float32, some updates amplify this (most of all AOFGD's (|θ| + ε)^(1−α) near θ = 0), so `05_…` compares the float32 difference with the Keras version's own float32 error. In float64, `tf.cast(0.9, tf.float64)` rounds through float32, so `05_…` converts Python numbers straight to float64 during the Keras float64 runs (the optimizer code is not touched).

## References

- <a id="ref-h1"></a>**[H1]** O. Herrera-Alcántara. Fractional Derivative Gradient-Based Optimizers for Neural Networks and Human Activity Recognition. *Applied Sciences* **12**(18), 9264 (2022). [doi:10.3390/app12189264](https://doi.org/10.3390/app12189264)
- <a id="ref-h2"></a>**[H2]** O. Herrera-Alcántara, J. R. Castelán-Aguilar. Fractional Gradient Optimizers for PyTorch: Enhancing GAN and BERT. *Fractal and Fractional* **7**(7), 500 (2023). [doi:10.3390/fractalfract7070500](https://doi.org/10.3390/fractalfract7070500)
- <a id="ref-r1"></a>**[R1]** S. Chen, C. Zhang, H. Mu. An Adaptive Learning Rate Deep Learning Optimizer Using Long and Short-Term Gradients Based on G–L Fractional-Order Derivative. *Neural Processing Letters* **56**, 106 (2024). [doi:10.1007/s11063-024-11571-7](https://doi.org/10.1007/s11063-024-11571-7)
- <a id="ref-r2"></a>**[R2]** X. Zhou, C. Zhao, Y. Huang. A Deep Learning Optimizer Based on Grünwald–Letnikov Fractional Order Definition. *Mathematics* **11**(2), 316 (2023). [doi:10.3390/math11020316](https://doi.org/10.3390/math11020316)
- <a id="ref-r3"></a>**[R3]** K. Xiang, C. Zhao, Q. Su, X. Zhou, J. Ye, Y. Huang. AOFGD: Adaptive Order Fractional Gradient Descent Method. SSRN preprint, not peer-reviewed (2025). [doi:10.2139/ssrn.5717167](https://doi.org/10.2139/ssrn.5717167) · [ssrn.com/abstract=5717167](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5717167)
- <a id="ref-c1"></a>**[C1]** R. Chipera, J. Du, I. Tsapara. Fractal and Chaotic Activation Functions in Echo State Networks: Preprocessing Topology Governs the Echo State Property. arXiv preprint (2025). [arXiv:2512.14675](https://arxiv.org/abs/2512.14675)
- <a id="ref-o1"></a>**[O1]** S. Raubitzek, G. Goldenits, S. Schrittwieser, P. König, K. Mallinger. Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks. *Fractal and Fractional* (2026, accepted). [arXiv:2608.14636](https://arxiv.org/abs/2608.14636) · [code](https://github.com/Raubkatz/FractalAndFractional2026)
- <a id="ref-o2"></a>**[O2]** S. Raubitzek, T. Kietreiber, S. Eresheim, A. Buchelt, K. Mallinger. Fractals in neural networks: Introducing a new class of activation functions. *Communications in Nonlinear Science and Numerical Simulation* **161**, 110064 (2026). [doi:10.1016/j.cnsns.2026.110064](https://doi.org/10.1016/j.cnsns.2026.110064) · [code](https://github.com/Raubkatz/Fractal_Activation_Functions)

**Grünwald–Letnikov background**

- <a id="ref-g1"></a>**[G1]** I. Podlubny. *Fractional Differential Equations.* Mathematics in Science and Engineering, vol. 198. Academic Press, San Diego (1999).
- <a id="ref-g2"></a>**[G2]** Y. Wei, Y. Chen, S. Cheng, Y. Wang. A Note on Short Memory Principle of Fractional Calculus. *Fractional Calculus and Applied Analysis* **20**, 1382–1404 (2017). [doi:10.1515/fca-2017-0073](https://doi.org/10.1515/fca-2017-0073)

<a id="base-optimizers"></a>**Base optimizers**

- <a id="ref-b1"></a>**[B1]** SGD: H. Robbins, S. Monro. A Stochastic Approximation Method. *Annals of Mathematical Statistics* **22**, 400–407 (1951). [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)
- <a id="ref-b2"></a>**[B2]** Momentum: B. T. Polyak. Some Methods of Speeding up the Convergence of Iteration Methods. *USSR Computational Mathematics and Mathematical Physics* **4**, 1–17 (1964). [doi:10.1016/0041-5553(64)90137-5](https://doi.org/10.1016/0041-5553(64)90137-5)
- <a id="ref-b3"></a>**[B3]** Nesterov momentum: Y. Nesterov. A method for solving the convex programming problem with convergence rate O(1/k²). *Soviet Mathematics Doklady* **27**, 372–376 (1983); in deep learning: I. Sutskever, J. Martens, G. Dahl, G. Hinton. On the importance of initialization and momentum in deep learning. *Proc. ICML*, PMLR **28**(3), 1139–1147 (2013). [proceedings.mlr.press/v28/sutskever13](https://proceedings.mlr.press/v28/sutskever13.html)
- <a id="ref-b4"></a>**[B4]** Adam and Adamax: D. P. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. *ICLR* (2015). [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- <a id="ref-b5"></a>**[B5]** AMSGrad: S. J. Reddi, S. Kale, S. Kumar. On the Convergence of Adam and Beyond. *ICLR* (2018). [openreview.net/forum?id=ryQu7f-RZ](https://openreview.net/forum?id=ryQu7f-RZ)
- <a id="ref-b6"></a>**[B6]** RMSprop: T. Tieleman, G. Hinton. Lecture 6.5 — RMSProp: Divide the gradient by a running average of its recent magnitude. *Neural Networks for Machine Learning*, Coursera (2012). [lecture slides](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- <a id="ref-b7"></a>**[B7]** Adadelta: M. D. Zeiler. ADADELTA: An Adaptive Learning Rate Method. arXiv (2012). [arXiv:1212.5701](https://arxiv.org/abs/1212.5701)
- <a id="ref-b8"></a>**[B8]** Adagrad: J. Duchi, E. Hazan, Y. Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. *Journal of Machine Learning Research* **12**, 2121–2159 (2011). [jmlr.org (PDF)](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
- <a id="ref-b9"></a>**[B9]** AdamW: I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. *ICLR* (2019). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- <a id="ref-b10"></a>**[B10]** Nadam: T. Dozat. Incorporating Nesterov Momentum into Adam. Stanford CS229 project report (2015); ICLR 2016 Workshop track. [report (PDF)](http://cs229.stanford.edu/proj2015/054_report.pdf) · [ICLR workshop paper (PDF)](https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf)

<a id="fractal-functions-mathematical-origins"></a>**Fractal functions: mathematical origins**

- <a id="ref-f1"></a>**[F1]** Weierstrass function: K. Weierstraß. Über continuirliche Functionen eines reellen Arguments, die für keinen Werth des letzteren einen bestimmten Differentialquotienten besitzen (lecture, 1872). Reprinted in R. Siegmund-Schultze (ed.), *Ausgewählte Kapitel aus der Funktionenlehre*, Teubner-Archiv zur Mathematik 9, Springer, Vienna (1988), pp. 190–193.
- <a id="ref-f2"></a>**[F2]** Blancmange (Takagi) function: T. Takagi. A Simple Example of the Continuous Function without Derivative. *Proc. Phys.-Math. Soc. Japan Ser. II* (Tokyo Sugaku-Butsurigakkwai Hokoku) **1**, F176–F177 (1903). [doi:10.11429/subutsuhokoku1901.1.F176](https://doi.org/10.11429/subutsuhokoku1901.1.F176)
- <a id="ref-f3"></a>**[F3]** Weierstrass–Mandelbrot function: M. V. Berry, Z. V. Lewis. On the Weierstrass–Mandelbrot Fractal Function. *Proc. R. Soc. Lond. A* **370**, 459–484 (1980). [doi:10.1098/rspa.1980.0044](https://doi.org/10.1098/rspa.1980.0044)
- <a id="ref-f4"></a>**[F4]** B. B. Mandelbrot. *The Fractal Geometry of Nature.* W. H. Freeman, San Francisco (1982).
- <a id="ref-f5"></a>**[F5]** Cantor set and Cantor function: G. Cantor. De la puissance des ensembles parfaits de points. *Acta Mathematica* **4**, 381–392 (1884).

**Software**

- <a id="ref-s1"></a>**[S1]** F. Chollet et al. Keras (2015–). [keras.io](https://keras.io) · [github.com/keras-team/keras](https://github.com/keras-team/keras)
- <a id="ref-s2"></a>**[S2]** M. Abadi et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems. arXiv (2016). [arXiv:1603.04467](https://arxiv.org/abs/1603.04467) · [tensorflow.org](https://www.tensorflow.org)
- <a id="ref-s3"></a>**[S3]** F. Pedregosa et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* **12**, 2825–2830 (2011). [jmlr.org/papers/v12/pedregosa11a](https://jmlr.org/papers/v12/pedregosa11a.html)
- <a id="ref-s4"></a>**[S4]** A. Paszke et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems* **32** (NeurIPS 2019), 8024–8035. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703) · [pytorch.org](https://pytorch.org)

**Data**

- <a id="ref-d1"></a>**[D1]** Iris data: R. A. Fisher. The use of multiple measurements in taxonomic problems. *Annals of Eugenics* **7**(2), 179–188 (1936). [doi:10.1111/j.1469-1809.1936.tb02137.x](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x) · OpenML dataset 61: [openml.org/d/61](https://www.openml.org/d/61)

### BibTeX

```bibtex
@article{herrera2022fractional,
  author  = {Herrera-Alc{\'a}ntara, Oscar},
  title   = {Fractional Derivative Gradient-Based Optimizers for Neural Networks and Human Activity Recognition},
  journal = {Applied Sciences},
  volume  = {12},
  number  = {18},
  pages   = {9264},
  year    = {2022},
  doi     = {10.3390/app12189264}
}

@article{herrera2023pytorch,
  author  = {Herrera-Alc{\'a}ntara, Oscar and Castel{\'a}n-Aguilar, Josu{\'e} R.},
  title   = {Fractional Gradient Optimizers for {PyTorch}: Enhancing {GAN} and {BERT}},
  journal = {Fractal and Fractional},
  volume  = {7},
  number  = {7},
  pages   = {500},
  year    = {2023},
  doi     = {10.3390/fractalfract7070500}
}

@article{chen2024adagl,
  author  = {Chen, Shuang and Zhang, Changlun and Mu, Haibing},
  title   = {An Adaptive Learning Rate Deep Learning Optimizer Using Long and Short-Term Gradients Based on {G--L} Fractional-Order Derivative},
  journal = {Neural Processing Letters},
  volume  = {56},
  number  = {2},
  pages   = {106},
  year    = {2024},
  doi     = {10.1007/s11063-024-11571-7}
}

@article{zhou2023gl,
  author  = {Zhou, Xiaojun and Zhao, Chunna and Huang, Yaqun},
  title   = {A Deep Learning Optimizer Based on {Gr{\"u}nwald--Letnikov} Fractional Order Definition},
  journal = {Mathematics},
  volume  = {11},
  number  = {2},
  pages   = {316},
  year    = {2023},
  doi     = {10.3390/math11020316}
}

@misc{xiang2025aofgd,
  author       = {Xiang, Kemeng and Zhao, Chunna and Su, Qian and Zhou, Xiaojun and Ye, Junjie and Huang, Yaqun},
  title        = {{AOFGD}: Adaptive Order Fractional Gradient Descent Method},
  howpublished = {SSRN preprint, abstract ID 5717167},
  year         = {2025},
  doi          = {10.2139/ssrn.5717167}
}

@misc{chipera2025fractal,
  author        = {Chipera, Rae and Du, Jenny and Tsapara, Irene},
  title         = {Fractal and Chaotic Activation Functions in Echo State Networks: Preprocessing Topology Governs the Echo State Property},
  year          = {2025},
  eprint        = {2512.14675},
  archivePrefix = {arXiv}
}

@article{raubitzek2026fractional,
  author  = {Raubitzek, Sebastian and Goldenits, Georg and Schrittwieser, Sebastian and K{\"o}nig, Philip and Mallinger, Kevin},
  title   = {Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks},
  journal = {Fractal and Fractional},
  year    = {2026},
  note    = {Accepted. Preprint: arXiv:2608.14636. Code: https://github.com/Raubkatz/FractalAndFractional2026}
}

@article{raubitzek2026fractals,
  author  = {Raubitzek, Sebastian and Kietreiber, Tobias and Eresheim, Sebastian and Buchelt, Alexander and Mallinger, Kevin},
  title   = {Fractals in neural networks: Introducing a new class of activation functions},
  journal = {Communications in Nonlinear Science and Numerical Simulation},
  volume  = {161},
  pages   = {110064},
  year    = {2026},
  doi     = {10.1016/j.cnsns.2026.110064}
}

@inproceedings{paszke2019pytorch,
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and K{\"o}pf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
  title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
  booktitle = {Advances in Neural Information Processing Systems 32},
  pages     = {8024--8035},
  year      = {2019}
}
```

## Citation

Please cite **the original publication of every optimizer and activation you use** (table above: Herrera-Alcántara [[H1]](#ref-h1), [[H2]](#ref-h2); Chen, Zhang & Mu [[R1]](#ref-r1); Zhou, Zhao & Huang [[R2]](#ref-r2); Xiang et al. [[R3]](#ref-r3); Chipera, Du & Tsapara [[C1]](#ref-c1)), our article [[O1]](#ref-o1) for the adaptive-memory and GL-memory optimizers, the experimental comparison and this platform, and [[O2]](#ref-o2) for the fractal activation functions. Full experiment code of the article: [github.com/Raubkatz/FractalAndFractional2026](https://github.com/Raubkatz/FractalAndFractional2026); code of the fractal activation paper: [github.com/Raubkatz/Fractal_Activation_Functions](https://github.com/Raubkatz/Fractal_Activation_Functions).
