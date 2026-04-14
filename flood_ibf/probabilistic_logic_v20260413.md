# Probabilistic Logic, Ensemble Storylines, and Bayesian Networks in High-Stakes Climate Services

**Version**: v20260413  
**Scope**: A conceptual audit of the Flood IBF BN we have built so far, asking whether it is genuinely *probabilistic logic* or a deductive rule-system dressed in Bayesian clothing, and where the method needs to evolve to support high-stakes anticipatory-action decisions.

---

## 1. The epistemic starting point

Operational climate services inherited two intellectual traditions that sit uneasily together:

1. **Frequentist hydrometeorology** — return periods, "1-in-100-year event" language, fixed thresholds derived from annual-maxima Gumbel fits or similar. The 2-year CMORPH threshold we use is a textbook example: 27 years of data, a Gumbel fit, a single deterministic number per pixel per duration. This framing is **deductive once the threshold is set**: if the observed or forecast accumulation exceeds T, the event is classified as "above 2-yr". The probability language ("2-yr return period") refers to long-run frequency, not to this specific event.

2. **Probabilistic forecasting from numerical weather prediction** — ECMWF IFS produces 51 ensemble members, each a physically self-consistent realization of the forecast atmosphere. Reading the ensemble as "51 possible worlds" is *modal* rather than frequentist: the ensemble is explicitly a sample from a distribution of possibilities, not an observed long-run frequency. This tradition is native to Bayesian reasoning — beliefs about the future given current observations.

Our pipeline stitches these two together: the frequentist threshold enters as an observation-based climatological reference, the ensemble enters as a forecast-based probability, and the BN tries to integrate them with expert judgment. This is not unusual — most operational IBF systems do the same thing — but the epistemic mixing is rarely examined, and it matters for what "probabilistic logic" can mean in our system.

---

## 2. What Bayesian networks actually do, versus what we want them to do

A Bayesian Network is, strictly speaking, two things at once:

- **A factorization of a joint probability distribution** via conditional independence structure. The DAG encodes which variables are conditionally independent given others.
- **A method for belief updating** — given evidence on some nodes, compute posterior distributions over others via Bayes' rule.

This machinery is genuinely probabilistic *only when the CPTs encode real uncertainty*. When the CPT is a sharp diagonal (e.g. `P(Monitor | Minimal) = 0.95`), the BN is doing deductive inference dressed in probabilistic notation — a case statement, essentially. The algebra of Bayes' rule still runs, but it's trivially collapsing because there's nothing to update.

Our current Flood IBF BN sits on this boundary in an uncomfortable way:

| Component | Probabilistic content | Real-world source |
|-----------|:---:|-------------------|
| `antecedent_rainfall` state | **None** (hard discretization) | 7-day IMERG sum → state by fixed mm cutoffs |
| `exceedance_prob` state | **Empirical** (p from 51 members) | Real frequency in ensemble |
| `tail_risk` state | **Semi-empirical** (pixel p95 of ens_max/threshold) | Quasi-possible-world reasoning |
| `rainfall_trend` state | **None** (regression + ±2 mm/day cut) | Deductive from IMERG |
| `risk_level` CPT | **Soft** (agreement mixing with uniform) but mostly rule-driven | Expert-written scenarios |
| `action` CPT | **Near-deterministic** (0.95 diagonal) | Hard policy mapping |
| `risk → action` inference | Trivial matmul | Lookup, not inference |

The honest conclusion: the BN as built is a **structured rule system with Bayesian plumbing**. The Bayesian plumbing matters for three reasons — auditability, explicit uncertainty bookkeeping, and the ability to upgrade to genuine probabilistic components later — but it does not by itself constitute probabilistic logic in the philosophically rich sense.

---

## 3. Deductive climate services versus probabilistic logic

Most operational EWS in the hydrometeorological domain are fundamentally deductive:

```
IF (forecast exceeds threshold T) AND (antecedent state in class S)
THEN issue warning of level W
```

The thresholds T are derived from long records; the rules from expert consensus. This is deduction from premises. The premises are informed by probability (return periods, ensemble spreads), but the inference step is classical.

**Probabilistic logic** (Nilsson 1986; Hailperin 1996) is a distinct approach. Instead of asking "is T exceeded?", it asks "what is the probability that a *particular proposition* is true given all evidence, and what decision does that probability license?" The logical operators (AND, OR, NOT) are extended to probabilities rather than truth values. A statement like "the 7-day forecast will cause a flood" carries a probability, not a truth value, and composite statements combine probabilities via Bayes or Dempster-Shafer rules.

In an operational flood-IBF context, probabilistic logic would mean:

- The ensemble is treated as a sample from the posterior predictive distribution, not as a frequency count.
- Evidence combines via conditional independence, with every CPT expressing genuine uncertainty about the conditional relationship.
- Decisions are framed as expected-utility maximizations under a loss function (cost-loss ratio for false alarms vs missed events), not as rule-based lookups.
- Uncertainty about thresholds, observations, and model skill propagates into the posterior — not just point estimates.

**Our system is not there yet.** It is a hybrid: frequentist thresholds feeding Bayesian-structured rules producing discrete categorical outputs. This is not a failure — it's a reasonable operational position — but naming it honestly matters for peer review, donor communication, and designing the next upgrade.

---

## 4. Ensemble prediction systems as possible worlds

A cleaner way to re-engage probabilistic logic for our use case is to take the ensemble seriously as a **possible-worlds structure** (in the sense of Kripke semantics or David Lewis's modal realism — adapted to forecasting):

- Each ECMWF ensemble member is a physically self-consistent simulation of a possible future.
- Together, the 51 members span a distribution of *what could happen*, weighted by initial-condition perturbations and stochastic physics.
- The ensemble **mean** averages across possible worlds — this is what P_heavy traditionally captures, and what our original pipeline fed to `exceedance_prob`.
- The ensemble **maximum at a given pixel** picks the most extreme world — this is what our `tail_risk` node now captures via ens_max_ratio.
- The ensemble **spread** is the model's admission of which worlds it thinks are possible.

The philosophical shift we made by adding `tail_risk` was (somewhat unintentionally) from *expected-value reasoning* toward *possible-worlds reasoning*. We stopped asking "on average, how likely is exceedance?" and started asking "does any plausible world lead to an exceedance we should care about?"

This matters because **for high-stakes, low-probability events, the expected value is the wrong decision variable**. The cost-loss asymmetry — a single missed flood event costs vastly more than many false alarms — means the tail of the distribution drives the optimal action. A BN that only sees the ensemble mean is blind to this. A BN that sees the tail (via ens_max, via hotspot_fraction, via the p95 upgrade) is tracking the decision-relevant possibility.

The Nairobi March 6 event is the case in point: the ensemble mean was ~18 mm (entirely benign), but at least one member projected 38 mm (boundary-mean) and some pixel combinations hit 131 mm. A mean-based system discards the tail; a possible-worlds system treats it as a live scenario.

---

## 5. Event-based storylines as a complementary epistemic tool

Climate storylines (Shepherd 2016, 2019; Trenberth et al. 2015; Zappa et al. 2021) are a deliberate alternative to probabilistic forecasting. Instead of producing a probability distribution over outcomes, a storyline provides:

- **A plausible, physically-coherent narrative** of what a specific event trajectory would look like.
- **Conditional reasoning**: "given that a MJO-phase-5-modulated cyclone approaches Mozambique, what are the impacts on Southern Tanzania?"
- **Explicit causality** rather than statistical correlation.

Storylines and ensemble forecasts are epistemic complements:

| Tool | Question answered | Strength | Weakness |
|------|-------------------|----------|----------|
| Ensemble mean | "What is the expected outcome?" | Calibrated long-run | Blind to tails |
| Ensemble max | "What is the worst plausible outcome?" | Captures tail | No likelihood |
| Return period threshold | "Is this a rare event?" | Frequentist grounding | Ignores current state |
| Storyline | "What would a specific bad event look like?" | Narrative traction for decision-makers | Subjective selection |
| BN with soft evidence | "How do multiple uncertain pieces combine?" | Structured uncertainty bookkeeping | Requires expert CPTs |

For **high-stakes anticipatory action**, the decision-maker often wants to know *"show me the world that scares me, and how likely is it?"*. Ensembles give the second part; storylines give the first. Our current pipeline provides neither at full fidelity:

- We do not currently pick out specific ensemble members as storylines.
- We collapse the 51-member distribution to summary statistics (mean, max, p95).
- The BN's output is categorical — it does not present scenarios.

A genuine integration would pipe **ensemble members through the BN individually** (not just their summary statistics), producing per-member risk-state distributions, and pick out the most informative members as "stories" — the member with highest `risk_level`, the median member, a member with unusual pathway. This is not a huge change technically, but it is a significant change philosophically: the BN becomes a **scenario evaluator**, not a summary-statistic digester.

---

## 6. Where the Flood IBF BN stands today — a critical self-assessment

### Strengths (what we've done well)
- **Tail-aware evidence**: `tail_risk` + pixel p95 aggregation explicitly captures possible-worlds reasoning on the forecast side.
- **Multi-evidence integration**: antecedent, trend, tail, spatial coverage, and exceedance mean are combined through a structured CPT, not a single threshold.
- **Auditable expert rules**: every rule in `compute_risk_probs` is human-readable and defensible.
- **Validated lead time**: the Nairobi Mar 4 **Moderate / Prepare** call is a 2-day early warning that a threshold-based system would have missed.
- **Full posterior exposed**: the output CSV contains all 5 risk probabilities and all 4 action probabilities — a decision-maker can reason over the full distribution, not just the argmax.

### Weaknesses (where we fall short of genuine probabilistic logic)
1. **Near-deterministic action CPT** — the `risk → action` CPT is a 0.95-diagonal case statement. It pretends to be probabilistic but it is a hard mapping. The honest architectural move is to remove the action node and compute the downstream label (CRMA state, traffic light) deterministically outside the BN, as [decision-output-riskassessment.md](decision-output-riskassessment.md) argues.
2. **No cost-loss framing** — we do not explicitly encode the asymmetric cost of false alarms vs missed events. The Act trigger at argmax could easily be re-cast as "trigger when `P(High) + P(Extreme) ≥ p*`, with `p* = C/L`" (the cost-loss ratio), which is the textbook probabilistic-logic formulation.
3. **Hard discretization of evidence** — antecedent rainfall at 59 mm is classified as "Wet", at 61 mm as "Very_Wet". A probabilistic-logic formulation would use soft evidence: a probability distribution over states reflecting the closeness to thresholds.
4. **Independent daily inference** — no temporal coupling; yesterday's posterior is not a prior for today. True continuity requires a Dynamic BN with a transition CPT on `risk_level`.
5. **No scenario-level output** — we aggregate across ensemble members before the BN. A probabilistic-logic treatment would pass each member through separately and aggregate afterwards.
6. **No explicit exposure / vulnerability** — our `risk_level` is really a **hazard likelihood**, not a full risk (= hazard × exposure × vulnerability per IPCC AR6 WGII). High-stakes anticipatory-action decisions need the latter.
7. **Expert CPTs have no learning loop** — they are frozen at writing time and never updated with observed flood outcomes. This violates the Bayesian principle of updating beliefs as evidence accumulates.

### Epistemic honesty

A candid label for our current BN would be:

> A **structured, auditable expert rule system with Bayesian-compliant probability accounting and tail-sensitive forecast ingestion**, operating as an evidence-driven risk-assessment engine with categorical output, intended to support — not replace — human decision-making in anticipatory action workflows.

This is a defensible scientific contribution. It is *not* a probabilistic-logic decision engine, and claiming so in peer review would invite correction. What we have is arguably better matched to current operational maturity than a fully probabilistic system would be: the expert CPTs carry forward practitioner knowledge that pure learning-from-data cannot match on small event databases, and the near-deterministic action mapping reflects the reality that operational decisions are governed by pre-agreed SOPs, not by runtime optimization.

---

## 7. Why probabilistic logic nevertheless matters, especially for anticipatory action

Anticipatory action is a high-stakes setting precisely because:

- **Early action has cost**: deployment of Red Cross resources, pre-positioning stockpiles, early disbursement of cash transfers.
- **Missed action has higher cost**: lives, livelihoods, downstream cascade into loss-and-damage frameworks.
- **False alarms erode trust**: a community that sees three Act-level triggers with no flood loses confidence in the fourth one, which happens to be real.

The cost-loss decision calculus is the textbook frame for this:

- Expected cost of acting when event does not occur = `C · P(¬event)`
- Expected loss of not acting when event occurs = `L · P(event)`
- Rational trigger: `Act iff P(event) ≥ C / L`

For anticipatory cash transfers, `C/L` is typically low (0.05–0.20) because acting-early is cheap and missing is devastating. This *matters*: it means the optimal trigger probability is far below 0.5. Our current system's argmax rule ("Act if P(Extreme) > all others") will tend to trigger at P ≥ 0.5, which is systematically too conservative for high-stakes settings. A probabilistic-logic formulation would explicitly encode the cost-loss ratio and trigger accordingly.

Storylines enter this frame as a complement: the cost-loss ratio gives the *decision threshold*, but it does not convey to decision-makers *what the bad world looks like*. Decision-makers often need a narrative to mobilize pre-positioned resources. A storyline — "this ensemble member shows Nairobi River peaking at level X by 3 AM on Mar 7 with Z mm rainfall overnight" — makes the probability concrete.

---

## 8. Where this leaves our method — a concrete forward path

Not all of the gaps are equally urgent or equally hard to close. Ordered by value × tractability:

### Near-term (within the current architecture)
1. **Remove the action node from the BN**, replace with an outside-BN deterministic rule on the risk posterior (per `decision-output-riskassessment.md`). This is a 30-line change. The BN becomes honestly Layer-1-only.
2. **Add cost-loss-based thresholds** for the CRMA state transitions. Expose them as `--cost-loss-ratio` CLI flags, with defaults informed by published FbF cost-benefit ratios (typically 0.1 for cash transfers, 0.2 for pre-positioned stockpiles).
3. **Emit per-member risk evaluations** as a sidecar CSV: for each of the 51 members, categorize its tail_risk independently and report the distribution of member-level risk levels. This starts giving us storyline material without changing the BN structure.

### Medium-term (within the current tooling)
4. **Soft evidence** via RxInfer.jl — feed in the probability-over-states rather than the hard-classified single state. Antecedent at 59 mm becomes `P(Wet) = 0.55, P(Very_Wet) = 0.45`, propagating the discretization uncertainty through the BN.
5. **Dynamic Bayesian Network** — add `P(risk_level[t] | risk_level[t-1])` transition CPT. Yesterday's Moderate elevates today's prior even if today's evidence is noisier. Proper temporal coupling.
6. **Storyline selection routine** — automated selection of 3 representative members per boundary (worst, median, best) and per-member BN traces, feeding storyline narratives.

### Longer-term (architectural)
7. **Replace expert CPTs with Dirichlet priors + learning loop** — use the historical flood record (EMDAT, FloodList, ICPAC incident database) to update the CPTs each season, keeping the expert rules as informative priors.
8. **Integrate exposure × vulnerability** — join population data (WorldPop), vulnerability indices (INFORM), and critical infrastructure layers into the risk calculation. This moves from hazard likelihood to full DRM risk per IPCC AR6 WGII.
9. **Hierarchical spatial BN** — share information across neighboring admin-1 regions via a Markov Random Field structure. A hotspot in Bungoma should elevate Busia's prior.

Each step moves the system deeper into genuine probabilistic logic. None is required to keep the current operational value; all are required to reach the level of rigor that donor-reviewed high-stakes anticipatory-action systems increasingly demand.

---

## 9. A positioning statement for peer review and donor communication

Draft language:

> "The East Africa Hazard Watch Flood IBF system combines analysis-ready cloud-hosted ensemble forecasts (ECMWF IFS, 51 members) with a Bayesian Network that integrates antecedent satellite observations (IMERG), pixel-level tail-risk signals (ensemble max against CMORPH 2-year return-period thresholds), and expert-elicited conditional rules to produce continuously-updated admin-1 risk assessments. The BN functions as a Layer-1 risk-assessment engine in the WMO Multi-Hazard Early Warning Systems framework — it performs auditable belief updating on evidence but does not replace the Layer-2 decision-authority, cost-loss reasoning, and institutional triggers that govern anticipatory action. Validation on the March 2026 Nairobi River flash-flood event shows that tail-risk-enhanced inference provides a 2-day early-warning signal invisible to ensemble-mean-based methods. Forthcoming extensions — cost-loss-based trigger thresholds, dynamic Bayesian temporal coupling, and per-member storyline evaluation — will move the system further toward the probabilistic-logic decision-support paradigm appropriate for high-stakes humanitarian applications."

This positioning is honest about what the system does and does not do, invites the right extensions, and does not overclaim probabilistic rigor that the near-deterministic action CPT does not support.

---

## 10. References and further reading

### Bayesian networks
- Jensen, F. V. (1996). *An Introduction to Bayesian Networks*. UCL Press.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.
- Kragt, M. E. et al. (2009). Bayesian Belief Networks for risk assessment. *Environmental Modelling & Software*, 24(10), 1197–1206.

### Probabilistic logic
- Nilsson, N. J. (1986). Probabilistic logic. *Artificial Intelligence*, 28(1), 71–87.
- Hailperin, T. (1996). *Sentential Probability Logic*. Lehigh University Press.
- Halpern, J. Y. (2003). *Reasoning About Uncertainty*. MIT Press.

### Climate services and IBF
- WMO (2015). *Multi-Hazard Early Warning Systems: A Checklist*. World Meteorological Organization.
- WMO (2021). *Guidelines on Multi-hazard Impact-based Forecast and Warning Services* (WMO-No. 1150).
- Hewitt, C., Mason, S., Walland, D. (2012). The Global Framework for Climate Services. *Nature Climate Change*, 2, 831–832.
- IPCC AR6 WGI Chapter 1, and WGII Chapter 1 (2022) on risk framing.

### Ensemble forecasting and decision theory
- Murphy, A. H. (1977). The value of climatological, categorical and probabilistic forecasts in the cost-loss ratio situation. *Monthly Weather Review*, 105, 803–816.
- Richardson, D. S. (2000). Skill and relative economic value of the ECMWF ensemble prediction system. *Quarterly Journal of the Royal Meteorological Society*, 126, 649–667.
- Katz, R. W., Murphy, A. H. (eds.) (1997). *Economic Value of Weather and Climate Forecasts*. Cambridge University Press.

### Storylines
- Shepherd, T. G. (2016). A common framework for approaches to extreme event attribution. *Current Climate Change Reports*, 2, 28–38.
- Shepherd, T. G. (2019). Storyline approach to the construction of regional climate change information. *Proceedings of the Royal Society A*, 475, 20190013.
- Zappa, G., Bevacqua, E., Shepherd, T. G. (2021). Improving climate change detection through optimal seasonal averaging. *Journal of Climate*, 34(23), 9269–9284.
- Trenberth, K. E., Fasullo, J. T., Shepherd, T. G. (2015). Attribution of climate extreme events. *Nature Climate Change*, 5, 725–730.

### Anticipatory action and forecast-based financing
- Coughlan de Perez, E. et al. (2015). Forecast-based financing: an approach for catalyzing humanitarian action based on extreme weather and climate forecasts. *Natural Hazards and Earth System Sciences*, 15, 895–904.
- IFRC (2021). *Anticipatory Action Framework*.
- Red Cross Red Crescent Climate Centre (2020). *Forecast-based Financing Manual*.
- Aven, T. (2016). Risk assessment and risk management: Review of recent advances. *European Journal of Operational Research*, 253(1), 1–13.

---

*This document is a conceptual companion to the operational pipeline documentation in `flood_bn_ibf_system_v20260412.md`. It does not specify code changes; it situates the work in the climate-services-epistemology literature.*
