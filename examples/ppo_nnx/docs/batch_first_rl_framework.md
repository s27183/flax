# Batch-First, Theory‑Guided RL: A Reverse‑Reasoning Framework

A practical, compute‑aware methodology for inventing and implementing RL algorithms by starting from the optimization‑time batch, then deriving estimators, upstream signals, and theory to support that batch. This document complements the PPO‑specific summary in [ppo_summary.md](./ppo_summary.md) and generalizes the approach.

---

## Why reverse reasoning?

Traditional RL often starts with a theory, then forces a data pipeline to match. We invert this:

1. Specify the immutable batch the optimizer consumes.
2. Derive unbiased/low‑variance estimators that use exactly those fields.
3. Work backwards to construct those fields reliably from rollouts or datasets.
4. Align the batch with neural architectures and hardware constraints.
5. Make the learning update a pure function of the batch and current parameters.

This keeps compute, memory, and data layout as first‑class constraints, encourages sufficient statistics in the batch, and invites new theory tailored to modern systems.

---

## Step 1: Specify the optimization‑time batch (the contract)

Define the minimal, sufficient fields the optimizer will see. Examples by family:

- Policy‑gradient / PPO‑like:
  - observations: $\mathbf{o}_{t}$
  - actions: $\mathbf{a}_{t}$
  - behavior log‑probabilities: $\log \pi_{\text{old}}(\mathbf{a}_{t}\mid \mathbf{o}_{t})$
  - advantages: $A_{t}$
  - returns or value targets: $R_{t}$
  - masks: $m_{t} \in \{0,1\}$ for truncations/dones
  - optional: value predictions at collection $V_{\text{old}}(\mathbf{o}_{t})$, KL/entropy weights per sample

- Off‑policy Q‑learning / actor‑critic:
  - $(\mathbf{o}_{t}, \mathbf{a}_{t}, r_{t}, \gamma_{t}, \mathbf{o}_{t+1}, d_{t})$
  - behavior stats: $\mu(\mathbf{a}_{t}\mid\mathbf{o}_{t})$ or $\log\mu$, importance weights $\rho_{t}$
  - n‑step / $\lambda$ targets, action masks

- Sequence/DT‑style:
  - tokenized sequences with padding masks, returns‑to‑go tokens, timesteps, action tokens

- Model‑based:
  - real and imagined short‑horizon tuples with uncertainty weights

Treat this as an API contract that remains stable across experiments.

---

## Step 2: From batch to estimators

Given the batch schema, derive loss components and gradient estimators that consume exactly those fields.

- Importance ratio for policy updates:

$$
  r_{t} \,=\, \exp\big( \log \pi_{\theta}(\mathbf{a}_{t}\mid\mathbf{o}_{t})
    - \log \pi_{\text{old}}(\mathbf{a}_{t}\mid\mathbf{o}_{t}) \big).
$$

- PPO clipped policy loss:

$$
  \mathcal{L}_{\pi}(\theta) \,=\, 
    \mathbb{E}\left[ \min\big( r_{t} A_{t},\ \mathrm{clip}(r_{t},\ 1-\varepsilon,\ 1+\varepsilon) A_{t} \big) \right].
$$

- Entropy bonus (for exploration):

$$
  \mathcal{L}_{\text{ent}}(\theta) \,=\, -\beta\, \mathbb{E}\left[\mathcal{H}\big(\pi_{\theta}(\cdot\mid\mathbf{o}_{t})\big)\right],
  \quad \mathcal{H}(p) = -\sum_{a} p(a)\,\log p(a).
$$

- Value loss (MSE or clipped):

$$
  \mathcal{L}_{V}(\theta) \,=\, \tfrac{1}{2}\,\mathbb{E}\big[ (V_{\theta}(\mathbf{o}_{t}) - R_{t})^{2} \big].
$$

  Clipped variant with stored $V_{\text{old}}$: $v_{\theta}^{\text{clip}} = \mathrm{clip}(V_{\theta} - V_{\text{old}}, -\varepsilon_{v}, +\varepsilon_{v}) + V_{\text{old}}$, then use $v_{\theta}^{\text{clip}}$ in place of $V_{\theta}$.

- Optional KL penalty or constraint (trust‑region flavor):

$$
  D_{\text{KL}}\big(\pi_{\text{old}}\,\Vert\,\pi_{\theta}\big)
  \approx \mathbb{E}\big[ \log \pi_{\text{old}}(\mathbf{a}_{t}\mid\mathbf{o}_{t}) 
    - \log \pi_{\theta}(\mathbf{a}_{t}\mid\mathbf{o}_{t}) \big].
$$

The total loss is a weighted sum with masks $m_{t}$ applied everywhere to exclude padded/truncated steps.

---

## Step 3: Upstream construction of signals

Work backwards from the estimators to compute each batch field from raw data.

- Generalized Advantage Estimation (GAE):

$$
  \delta_{t} \,=\, r_{t} + \gamma (1 - d_{t}) V(\mathbf{o}_{t+1}) - V(\mathbf{o}_{t}),\quad
  A_{t} \,=\, \sum_{\ell=0}^{T-1-t} (\gamma\lambda)^{\ell} \, \delta_{t+\ell} \, \prod_{j=0}^{\ell-1} (1-d_{t+j}).
$$

  Return / value target via $\lambda$-return: $R_{t}^{(\lambda)} = A_{t} + V(\mathbf{o}_{t})$.

- Advantage normalization (per batch or per device):

$$
  \hat{A}_{t} \,=\, \frac{A_{t} - \mu_{A}}{\sigma_{A} + \epsilon}.
$$

- Importance weights for off‑policy data (if used):

$$
  \rho_{t} \,=\, \frac{\pi_{\theta}(\mathbf{a}_{t}\mid\mathbf{o}_{t})}{\mu(\mathbf{a}_{t}\mid\mathbf{o}_{t})},\quad
  \text{ESS} \,=\, \frac{\big(\sum_{t} w_{t}\big)^{2}}{\sum_{t} w_{t}^{2}} \quad (w_{t}=\rho_{t}\,\text{or clipped}).
$$

- Masks: $m_{t} = 1 - d_{t}$ for true terminations; include separate timeout masks if episodes truncate due to time limits.

- Log‑probabilities: store $\log \pi_{\text{old}}$ at action time or guarantee recomputation under a frozen copy of the acting parameters.

---

## Step 4: Align the batch with models and hardware

- Shapes and layout: collect $T$ steps across $N$ envs, yielding $(T, N, \cdot)$; flatten to $B = T\times N$ for SGD minibatches, or keep time‑major for sequence models with attention masks.
- Stabilize XLA/JIT: prefer static shapes; shard $B$ evenly across devices; use bf16/fp16 for matmuls with fp32 accumulators for statistics.
- Memory budget: decide whether to store $\log \pi_{\text{old}}$, $V_{\text{old}}$ for all $(T,N)$ or recompute; avoid recompiles by keeping batch schemas and dtypes fixed.

---

## Step 5: Make the update a pure function of the batch

Ensure the optimizer step is fully determined by the input batch and current parameters:

- Determinism and RNG scoping (e.g., dropout keyed by step).
- No hidden cross‑step state; if you need accumulators, store them explicitly.
- Clear collector ↔ optimizer separation using the batch contract as the only interface.

This drastically improves reproducibility and debuggability.

---

## Reusable patterns in batch‑first design

- Advantage‑like signals with per‑batch centering and scaling.
- Ratios and reweighting: $r_{t}$ histograms, clip fraction monitoring.
- Baselines: learned critics; generalized baselines for off‑policy (V‑trace, Retrace).
- Bootstrapping and $\lambda$‑smoothing: TD($\lambda$), GAE.
- Masks everywhere: episode ends, timeouts, padding; rigorous masking in losses and stats.
- Uncertainty and conservatism: ensembles/dropout for model‑based; conservative Q for offline.
- Regularization: entropy, KL to behavior, weight decay, target networks.
- Calibration: value clipping and predicted‑vs‑realized returns diagnostics.

---

## Worked example: PPO derived from the batch

Target batch (per sample):
$(\mathbf{o}_{t}, \mathbf{a}_{t}, \log \pi_{\text{old}}(\mathbf{a}_{t}\mid\mathbf{o}_{t}), A_{t}, R_{t}, m_{t}, V_{\text{old}}(\mathbf{o}_{t})\,\text{(optional)})$.

Losses:

$$
\begin{aligned}
\mathcal{L}_{\pi}(\theta) &= \mathbb{E}\left[ \min\big( r_{t} A_{t},\ \mathrm{clip}(r_{t}, 1-\varepsilon, 1+\varepsilon) A_{t} \big) \right],\\
\mathcal{L}_{V}(\theta) &= \tfrac{1}{2}\,\mathbb{E}\big[ (V_{\theta}(\mathbf{o}_{t}) - R_{t})^{2} \big]\ \text{(or clipped)},\\
\mathcal{L}_{\text{ent}}(\theta) &= -\beta\, \mathbb{E}\big[\mathcal{H}(\pi_{\theta}(\cdot\mid\mathbf{o}_{t}))\big],\\
\mathcal{L}_{\text{KL}}(\theta) &= \alpha\, \mathbb{E}\big[ \log \pi_{\text{old}}(\mathbf{a}_{t}\mid\mathbf{o}_{t}) - \log \pi_{\theta}(\mathbf{a}_{t}\mid\mathbf{o}_{t}) \big].
\end{aligned}
$$

Total: $\mathcal{L}= -\mathcal{L}_{\pi} + c_{v}\,\mathcal{L}_{V} + \mathcal{L}_{\text{ent}} + \mathcal{L}_{\text{KL}}$, with masks $m_{t}$ applied to all expectations.

Upstream signals via GAE as given above; normalize advantages per batch; flatten $(T,N)$ to $B$, shuffle, and run $K$ epochs of mini‑batch SGD.

For an end‑to‑end PPO data/process view, see [ppo_summary.md](./ppo_summary.md).

---

## Offline and model‑based variants from the same lens

- Offline actor‑critic (e.g., IQL/AWR‑style):
  - Batch: $(\mathbf{o},\mathbf{a},r,\gamma,\mathbf{o}',d, m)$, optional behavior $\log\mu$.
  - Critic: expectile regression to reduce overestimation; policy: advantage‑weighted regression with weights $w=\exp(A/\beta)$ (clipped).
  - No high‑variance IS required; conservatism controlled by expectile $\tau$ and temperature $\beta$.

- Short‑horizon model‑based improvement:
  - Batch mixes real and imagined tuples with per‑sample uncertainty weights $u\in[0,1]$.
  - Losses consume both, down‑weighting high‑uncertainty imagined states: multiply per‑sample terms by $u$.

---

## How theory connects to the batch contract

Treat “what must be in the batch” as specifying sufficient statistics for your estimator, then pose a constrained optimization:

$$
\min_{\text{estimator}} \; \mathbb{E}[\ell]\ \text{s.t.}\ \operatorname{Var}(\nabla \ell) \leq \sigma^{2},\ \text{bias}\leq b,\ \text{compute/memory}\leq C.
$$

This can yield batch‑dependent innovations, e.g.:

- Per‑sample adaptive clipping $\varepsilon_{i}$ based on variance proxies $\hat{\sigma}_{g,i}$: choose $\varepsilon_{i} \propto 1/(\hat{\sigma}_{g,i}+\tau)$.
- KL‑targeting via a penalty $\alpha$ tuned to hit a batch‑measured target $\bar{D}_{\text{KL}}$: update $\alpha \leftarrow \alpha \cdot \exp(\eta(\bar{D}_{\text{KL}} - D_{\text{target}}))$.
- Value‑aware clipping: attenuate policy updates for large $|A_{t}|$ or high epistemic uncertainty (requires an uncertainty field in the batch).

---

## Diagnostics and health checks

- Batch integrity:
  - Shapes, dtypes, masks consistent; no gradients through masked tokens.
  - Advantage mean $\approx 0$ after normalization; track $\operatorname{std}(A)$.
  - Returns vs. predicted values: calibration plots; monitor clipped fraction for value loss.
- Estimator sanity:
  - Ratio $r_{t}$ histogram centered near $1$; healthy clip fraction (e.g., 5–30% depending on $\varepsilon$).
  - Entropy decays as performance increases without collapsing prematurely.
  - Off‑policy: effective sample size (ESS) reasonable; clipped IS bounded.
- Systems:
  - Static shapes for JIT; no recompiles across steps.
  - High throughput; no dataloader/collector stalls.
- Ablations:
  - Remove each stabilization trick (adv norm, clip, entropy) and verify expected degradations.
  - Vary $\lambda,\gamma,\varepsilon$; sensitivity is bounded.

---

## Suggested next steps

- Formalize batch contract templates for the RL families you care about.
- For one algorithm (e.g., PPO), implement the optimizer as a pure function of the batch; confirm there are no hidden dependencies.
- Add batch‑level diagnostics reflecting your sufficiency assumptions; evolve the contract or estimator if assumptions break.
- Explore one innovation unlocked by an added batch field (e.g., per‑sample variance, behavior KL, uncertainty).
- Document compute budgets alongside the batch (memory per device, steps/sec) to keep designs grounded.

---

## Closing thought

Reverse reasoning reframes RL as: choose the sufficient statistics we can optimize under constraints, then derive the best estimator using them. This aligns math with data and compute, and opens a path to new theory native to modern hardware and software stacks.
