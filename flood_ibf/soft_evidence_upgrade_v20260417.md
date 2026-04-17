# Soft-Evidence Bayesian Inference for Flood IBF: Four Upgrades and Their Theoretical Foundations

**Version**: v20260417
**Scope**: Documents the four code-level changes introduced in commits `18ad13b` and `dec0225` on `jua-bnet`, explains each through the lens of peer-reviewed theory, and identifies which aspects are novel contributions versus established methodology applied to a new domain.

---

## Overview of changes

| # | Change | Commit | Lines |
|---|--------|--------|------:|
| 1 | RxInfer.jl as the required inference engine with multi-parent tensor CPT | `18ad13b` | ~300 |
| 2 | Virtual-evidence channels for soft (probabilistic) parent observations | `18ad13b` | ~80 |
| 3 | Gaussian soft-binning of continuous evidence at discretisation boundaries | `dec0225` | ~60 |
| 4 | Pencil-chunked zarr data store for per-pixel-member ensemble access | `dec0225` | ~30 |

Together these implement **upgrade #4** from the 9-point forward path in `probabilistic_logic_v20260413.md` and enable the BN to propagate discretisation uncertainty rather than silently collapsing it at hard bin edges.

---

## 1. From hand-rolled matrix lookup to library-native message passing

### What changed

The inference path in `flood_bn_ibf_v1.jl` was a hand-rolled column lookup on a pre-computed `(5 x N_combos)` CPT matrix:

```julia
risk_probs = risk_cpt[:, parent_idx]        # one column, O(1)
action_probs = action_cpt * risk_probs      # 4x5 * 5, O(20)
```

This was replaced by an RxInfer.jl `@model` with explicit `Categorical` parent nodes, `DiscreteTransition` factor nodes carrying a multi-dimensional tensor CPT, and `diageye(K)` virtual-evidence observation channels — all evaluated via the library's sum-product message passing.

### Theoretical basis

**Factor graphs and the sum-product algorithm.** A Bayesian Network's joint distribution `P(X1, ..., Xn)` factors over its DAG into a product of conditional distributions. The sum-product algorithm (Kschischang, Frey & Loeliger, 2001) computes exact marginals by passing messages along the edges of the corresponding factor graph. For a tree-structured graph (or a single-pass DAG like ours), one round of message passing yields exact posteriors — equivalent to variable elimination but expressed in a form that naturally handles:

- **Partial evidence** (some parents observed, others latent)
- **Soft evidence** (observation is a distribution, not a point)
- **Streaming updates** (new evidence arrives asynchronously)

None of these are possible with a hard column lookup. The matmul path and the message-passing path produce numerically identical results under hard one-hot evidence (validated: max |Δ| = 1.49e-9 over 2,270 boundary-days), but only the message-passing path generalises.

**RxInfer.jl specifically.** RxInfer (Bagaev & de Vries, 2023) implements reactive message passing on factor graphs via the ReactiveMP.jl engine. Its `DiscreteTransition` node natively supports higher-order tensor CPTs (verified up to 6 dimensions / 5 conditioning parents in this work), and its `@model` macro compiles the graph structure once for reuse across data points. First-call latency is ~30 s (JIT + graph compilation); subsequent calls ~1 s per boundary.

### Key references

- Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). Factor graphs and the sum-product algorithm. *IEEE Transactions on Information Theory*, 47(2), 498–519.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann. — Chapter 4: message passing on polytrees.
- Bagaev, D. & de Vries, B. (2023). RxInfer: A Julia package for reactive real-time Bayesian inference. *Journal of Open Source Software*, 8(84), 5161.
- Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press. — Chapters 9–10: exact inference via variable elimination and belief propagation.

---

## 2. Virtual evidence for soft parent observations

### What changed

Each parent node in the BN now has a paired "observation channel" — a child node connected through an identity-matrix `DiscreteTransition`:

```julia
ant      ~ Categorical(fill(1/5, 5))           # uninformative prior
ant_data ~ DiscreteTransition(ant, diageye(5))  # virtual-evidence channel
```

When hard evidence is available, `ant_data` receives a one-hot vector `[0,0,1,0,0]`. When soft evidence is available, it receives a probability vector `[0.0, 0.1, 0.55, 0.35, 0.0]`. The same model handles both cases without structural change — the difference is entirely in the data.

### Theoretical basis

**Virtual evidence (Pearl, 1988; Bilmes, 2004; Chan & Darwiche, 2005).** Pearl introduced the distinction between "hard evidence" (a variable is known to be in state `k`) and "virtual evidence" (also called "likelihood evidence"): the observation is characterised by a likelihood ratio vector `λ(x) = [λ_1, ..., λ_K]` that modulates the prior. A variable `X` with prior `P(X)` and virtual evidence `λ` has posterior:

```
P(X = k | λ) ∝ P(X = k) · λ_k
```

When `λ` is one-hot, this reduces to hard evidence. When `λ` is a probability vector, it acts as a soft constraint.

The `diageye(K)` observation channel implements this exactly: `ant_data ~ DiscreteTransition(ant, I_K)` means `P(ant_data = j | ant = k) = δ_{jk}`. Observing `ant_data = p_vec` sends a message proportional to `p_vec` backward to `ant`, which is precisely Pearl's virtual-evidence semantics.

**Why this matters for discretised continuous variables.** In expert-elicited BNs with continuous parent variables that are discretised into categorical states, observations near a bin boundary carry inherent ambiguity. A 7-day rainfall total of 59.5 mm falls just below the "Wet" threshold (60 mm) and is classified as "Normal" — but its true state is uncertain. Virtual evidence lets the system represent this as `P(Normal) = 0.52, P(Wet) = 0.48` rather than forcing a binary choice.

Uusitalo (2007) and Marcot (2012) both identified discretisation artefacts as a major source of error in environmental BNs, and recommended continuous or semi-continuous alternatives. Virtual evidence on the existing discrete structure achieves a similar effect without restructuring the DAG — the discretisation uncertainty is absorbed by the evidence channel rather than the CPT.

### Key references

- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann. — Section 2.2: virtual evidence and likelihood weighting.
- Bilmes, J. A. (2004). On virtual evidence and soft evidence in Bayesian networks. *Technical Report UWEETR-2004-0016*, University of Washington.
- Chan, H. & Darwiche, A. (2005). On the revision of probabilistic beliefs using uncertain evidence. *Artificial Intelligence*, 163(1), 67–90.
- Uusitalo, L. (2007). Advantages and challenges of Bayesian networks in environmental modelling. *Ecological Modelling*, 203(3–4), 312–318.
- Marcot, B. G. (2012). Metrics for evaluating performance and uncertainty of Bayesian network models. *Ecological Modelling*, 230, 50–62.
- Neil, M., Fenton, N., & Nielson, L. (2000). Building large-scale Bayesian networks. *The Knowledge Engineering Review*, 15(3), 257–284. — Discusses discretisation bias in hybrid BNs.

---

## 3. Gaussian soft-binning of continuous evidence

### What changed

A new function `soft_bin(x, node, sigma)` in `flood_data_prep.py` converts each continuous observation into a probability vector over the node's discrete states using a Gaussian kernel:

```python
P(state_k) = Φ((upper_k - x) / σ) - Φ((lower_k - x) / σ)
```

where `Φ` is the standard normal CDF, `[lower_k, upper_k]` are the bin edges (mirroring the Julia `categorize_*` cutoffs), and `σ` is a per-node bandwidth parameter set to ~30% of the narrowest bin spacing.

The 20 resulting probability columns (`ant_p1..p5`, `exc_p1..p5`, `spa_p1..p3`, `trn_p1..p3`, `tail_p1..p4`) are appended to the daily CSV when `--soft-evidence` is passed, and Julia's `run_csv` auto-detects and feeds them as virtual evidence.

### Theoretical basis

**Gaussian membership functions and fuzzy classification.** The soft-binning operation is mathematically equivalent to computing the probability that a Gaussian-distributed random variable falls in each bin. This has two distinct theoretical lineages:

1. **Bayesian measurement error.** If the true value `X*` is observed with Gaussian noise `X = X* + ε`, `ε ~ N(0, σ²)`, then `P(X* ∈ bin_k | X = x) ∝ ∫_{bin_k} N(x; μ, σ²) dμ`. This is the standard Bayesian treatment of measurement uncertainty in a discretised domain (Gelman et al., 2013, §5.5). The `σ` parameter should ideally be set from the physical measurement noise — IMERG retrieval uncertainty (~10% for daily totals per Huffman et al., 2020), ensemble sampling variability (`√(p(1-p)/51)` for a 51-member ensemble), or Gumbel-fit posterior width.

2. **Fuzzy sets (Zadeh, 1965).** The Gaussian CDF integration is equivalent to computing the degree of membership in each fuzzy set defined by a Gaussian membership function. Fuzzy-logic approaches to environmental classification use exactly this operation (Silvert, 2000; Adriaenssens et al., 2004). The bandwidth `σ` plays the role of the fuzzification parameter.

3. **Kernel density estimation (Silverman, 1986).** The Gaussian kernel `K_σ(x - c_k)` evaluated at each bin's centre is a KDE-like operation, but our formulation integrates the kernel over each bin rather than evaluating at a point, making it a proper probability assignment that sums to 1.

**Relationship to the σ → 0 limit.** As `σ → 0`, the Gaussian CDF step function collapses to a Heaviside function and `soft_bin` reproduces the hard classification exactly. This was verified: when `σ` is small relative to the distance from `x` to all bin edges, the output is effectively one-hot. The 10-day validation confirmed that for 2,249 of 2,270 boundary-days the hard and soft results agree — the 21 differences occur precisely at boundary-days where `x` is within ~2σ of a bin edge.

### Empirical result

Over the Mar 1–10 window (227 admin-1 × 10 days = 2,270 boundary-days), soft evidence:
- Promoted **14 boundary-days** from Monitor → Evaluate (previously invisible near-threshold cases).
- Demoted **5 boundary-days** from Assess → Evaluate (softened evidence pulling probability from Moderate toward Low).
- Escalated **1 boundary-day** from Assess → Actionable_Risk (Marsabit Mar 1, where tail-risk soft evidence crossed the cost-loss threshold).
- Created **1 net new CRMA tier**: Evaluate went from 0 to 19 boundary-days — a tier that was structurally unreachable under hard classification because the cost-loss thresholds were always jumped over rather than gradually approached.

### Key references

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall/CRC. — Section 5.5: measurement error models.
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.
- Adriaenssens, V., De Baets, B., Goethals, P. L. M., & De Pauw, N. (2004). Fuzzy rule-based models for decision support in ecosystem management. *Science of the Total Environment*, 319(1–3), 1–12.
- Silvert, W. (2000). Fuzzy indices of environmental conditions. *Ecological Modelling*, 130(1–3), 111–119.
- Huffman, G. J. et al. (2020). Integrated Multi-satellite Retrievals for the Global Precipitation Measurement (GPM) mission (IMERG). *Satellite Precipitation Measurement*, Vol. 1, Springer, 343–353.

---

## 4. Pencil-chunked zarr store for per-pixel-member ensemble access

### What changed

A `--pencil` flag on `flood_data_prep.py` switches the ECMWF read from the icechunk store (chunks `(1, 1, 53, 157, 145)` — one member × full grid, "pancake") to a parallel zarr mirror (chunks `(1, 51, 53, 16, 16)` — all 51 members × all leads for a 16×16 spatial tile, "pencil").

Benchmark on this host (D = 2026-03-06):

| Access pattern | Pancake (icechunk) | Pencil (zarr) | Winner |
|---|---:|---:|---|
| Full init slice (51×53×157×145) | **11.8 s** | 52.2 s | Pancake 4.4× |
| Single pixel × 51 members × 53 leads | 9.3 s | **1.4 s** | Pencil 6.5× |

The current zonal-statistics pipeline uses the full-grid pattern (pancake wins). The pencil store earns its keep when future upgrades (per-pixel Gumbel-threshold integration, per-member storyline traces) shift the access pattern to pixel-time-series.

### Theoretical basis

**Analysis-ready cloud-optimised (ARCO) data and chunk layout theory.** The choice between "pencil" (narrow spatial, deep member/time) and "pancake" (full spatial, single member/time) chunk layouts is a well-studied design trade-off in geoscientific array storage (Abernathey et al., 2021; Stern et al., 2022). The optimal layout depends on the dominant access pattern:

- **Pancake** (space-major): ideal when the query is "give me the full spatial field for one member" — typical of map rendering, zonal statistics, and spatial aggregation. Each chunk contains a full field; one I/O per member.
- **Pencil** (member-major or time-major): ideal when the query is "give me all members (or all time steps) at one pixel" — typical of per-pixel ensemble analysis, time-series diagnostics, and member-level storyline extraction.

Maintaining both layouts as parallel views of the same data (a "virtual rechunking" or "multi-resolution store") is the recommended approach in ARCO architectures (Abernathey et al., 2021). Our implementation uses two physical stores (icechunk for pancake, zarr for pencil) rather than virtual rechunking, trading storage duplication for simplicity.

**Connection to the soft-evidence upgrade.** The pencil store's value proposition is directly tied to upgrade #4's deeper path: per-pixel-member soft evidence requires reading `accum[member, pixel]` for all 51 members at each pixel, then integrating against the Gumbel threshold distribution at that pixel. This is a pure pencil access pattern — one chunk read per 16×16 tile delivers all members and leads, compared to 51 chunk reads under the pancake layout.

### Key references

- Abernathey, R. P. et al. (2021). Cloud-native repositories for big scientific data. *Computing in Science & Engineering*, 23(2), 26–35.
- Stern, C. et al. (2022). Pangeo Forge: Crowdsourcing analysis-ready, cloud-optimized open data. *Frontiers in Climate*, 3, 782909.
- Hoyer, S. & Hamman, J. (2017). xarray: N-D labeled arrays and datasets in Python. *Journal of Open Research Software*, 5(1), 10.
- Miles, A. et al. (2024). Zarr: An open standard for cloud-optimized multidimensional arrays. *Zenodo*. doi:10.5281/zenodo.10790680

---

## What is novel versus established methodology

| Aspect | Status | Notes |
|--------|--------|-------|
| Message passing on discrete BNs | **Established** | Textbook (Pearl 1988, Koller & Friedman 2009) |
| Virtual evidence | **Established** | Pearl 1988, formalised by Chan & Darwiche 2005 |
| Gaussian soft-binning of BN evidence | **Established in fuzzy-logic literature** (Zadeh 1965, Silvert 2000); **novel application** to IBF/EWS with cost-loss triggers |
| RxInfer.jl multi-parent tensor CPT | **Established library feature** (Bagaev & de Vries 2023); **novel use** in an operational hydrometeorological BN with 5 conditioning parents |
| Pencil vs pancake chunk layout | **Established in ARCO** (Abernathey et al. 2021); **novel pairing** with per-pixel ensemble-threshold soft evidence |
| Cost-loss triggers sensitive to soft evidence | **Theoretically anticipated** (Murphy 1977, Richardson 2000, Lopez et al. 2020); **first empirical demonstration** (to our knowledge) in an operational IBF system showing that Gaussian soft-binning reveals 19 otherwise-invisible Evaluate boundary-days |

The primary novel contribution is the **integration**: applying virtual evidence, Gaussian soft-binning, and cost-loss triggers together in an operational anticipatory-action pipeline, and demonstrating empirically that discretisation uncertainty — previously invisible — changes decisions at the margin. The individual components are well-established; their combination in this specific operational context has not been previously reported.

---

## References (consolidated)

### Bayesian networks and inference
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann.
- Jensen, F. V. (1996). *An Introduction to Bayesian Networks*. UCL Press.
- Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.
- Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). Factor graphs and the sum-product algorithm. *IEEE Transactions on Information Theory*, 47(2), 498–519.

### Virtual and soft evidence
- Bilmes, J. A. (2004). On virtual evidence and soft evidence in Bayesian networks. *Technical Report UWEETR-2004-0016*, University of Washington.
- Chan, H. & Darwiche, A. (2005). On the revision of probabilistic beliefs using uncertain evidence. *Artificial Intelligence*, 163(1), 67–90.

### RxInfer.jl
- Bagaev, D. & de Vries, B. (2023). RxInfer: A Julia package for reactive real-time Bayesian inference. *Journal of Open Source Software*, 8(84), 5161.

### Discretisation and environmental BNs
- Uusitalo, L. (2007). Advantages and challenges of Bayesian networks in environmental modelling. *Ecological Modelling*, 203(3–4), 312–318.
- Marcot, B. G. (2012). Metrics for evaluating performance and uncertainty of Bayesian network models. *Ecological Modelling*, 230, 50–62.
- Neil, M., Fenton, N., & Nielson, L. (2000). Building large-scale Bayesian networks. *The Knowledge Engineering Review*, 15(3), 257–284.
- Kragt, M. E. et al. (2009). Bayesian Belief Networks for risk assessment. *Environmental Modelling & Software*, 24(10), 1197–1206.

### Fuzzy sets and kernel methods
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.
- Adriaenssens, V. et al. (2004). Fuzzy rule-based models for decision support in ecosystem management. *Science of the Total Environment*, 319(1–3), 1–12.
- Silvert, W. (2000). Fuzzy indices of environmental conditions. *Ecological Modelling*, 130(1–3), 111–119.

### Measurement uncertainty and Bayesian data analysis
- Gelman, A. et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall/CRC.
- Huffman, G. J. et al. (2020). Integrated Multi-satellite Retrievals for GPM (IMERG). *Satellite Precipitation Measurement*, Vol. 1, Springer, 343–353.

### Cost-loss decision theory and anticipatory action
- Murphy, A. H. (1977). The value of climatological, categorical and probabilistic forecasts in the cost-loss ratio situation. *Monthly Weather Review*, 105, 803–816.
- Richardson, D. S. (2000). Skill and relative economic value of the ECMWF ensemble prediction system. *QJRMS*, 126, 649–667.
- Lopez, A., Coughlan de Perez, E., Bazo, J., Suarez, P., van den Hurk, B., & van Aalst, M. (2020). Bridging forecast verification and humanitarian decisions. *Weather and Climate Extremes*, 27, 100167.
- Coughlan de Perez, E. et al. (2015). Forecast-based financing. *NHESS*, 15, 895–904.

### Cloud-optimised geoscientific data
- Abernathey, R. P. et al. (2021). Cloud-native repositories for big scientific data. *Computing in Science & Engineering*, 23(2), 26–35.
- Stern, C. et al. (2022). Pangeo Forge: Crowdsourcing analysis-ready, cloud-optimized open data. *Frontiers in Climate*, 3, 782909.
- Miles, A. et al. (2024). Zarr: An open standard for cloud-optimized multidimensional arrays. *Zenodo*.

### Probabilistic logic (background)
- Nilsson, N. J. (1986). Probabilistic logic. *Artificial Intelligence*, 28(1), 71–87.
- Hailperin, T. (1996). *Sentential Probability Logic*. Lehigh University Press.
- Halpern, J. Y. (2003). *Reasoning About Uncertainty*. MIT Press.

---

*This document is a companion to `probabilistic_logic_v20260413.md` (conceptual audit) and `flood_bn_ibf_system_v20260412.md` (system technical documentation). It focuses specifically on the theoretical grounding of the four code changes introduced in April 2026.*
