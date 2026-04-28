#=
Drought Impact-Based Forecasting using Bayesian Networks — RxInfer.jl

Drought analogue of bn-ibf/flood_ibf/flood_bn_ibf_v1.jl. Same BN topology
(5 parents → risk_level → action), same RxInfer message-passing engine,
same direct-matmul + tensor-contraction fallback, same CRMA cost-loss
decision rule, same DBN temporal coupling and per-member storyline picker.

Only the parent semantics, bin cutoffs, and CPT stress weights are
domain-specific (drought-side: SPI ≤ RP rather than TP ≥ RP).

Reads the CSV from drought_data_prep.py:
    id, name, country,
    current_spi3, current_spi3_category,
    spi3_trend, trend_slope_spi_per_month,
    forecast_deficit_prob, deficit_prob_lead1,
    spatial_coverage, ...,
    forecast_agreement,
    ens_min_spi (p5 of ens-min across leads), ...,
    target_date,
    [optional cur_p1..cur_p5, def_p1..def_p5, spa_p1..spa_p3,
     trn_p1..trn_p3, tail_p1..tail_p4]

Usage:
    julia --project drought_bn_ibf_v1.jl \
        --input-csv  bn_inputs/drought_inputs_2026-04-01.csv \
        --output-csv output/drought_bn_v1_2026-04-01.csv
    julia --project drought_bn_ibf_v1.jl --test

Author: ICPAC IBF Team
Date: April 2026
=#

using LinearAlgebra
using Printf
using CSV
using DataFrames
using RxInfer

# ============================================================================
# CONSTANTS
# ============================================================================

# State labels — drought-side mapping. Order is increasing-stress for the
# numerical CPT logic (see compute_risk_probs).
const CURRENT_SPI3_STATES = ["Above_Normal", "Normal", "Mild_Drought",
                              "Moderate_Drought", "Severe_Drought"]            # 5
const DEFICIT_STATES      = ["Very_Low", "Low", "Medium", "High", "Very_High"] # 5
const SPATIAL_STATES      = ["Localized", "Moderate", "Widespread"]            # 3
const TREND_STATES        = ["Improving", "Stable", "Deteriorating"]           # 3
const AGREEMENT_STATES    = ["Low", "Medium", "High"]                          # 3
const TAIL_RISK_STATES    = ["Nil", "Low", "Moderate", "High"]                 # 4
const RISK_STATES         = ["Minimal", "Low", "Moderate", "High", "Extreme"]  # 5
const ACTION_STATES       = ["Monitor", "Alert", "Prepare", "Act"]             # 4 (deprecated)
const CRMA_STATES         = ["Monitor", "Evaluate", "Assess", "Actionable_Risk"] # 4
const TRAFFIC_LIGHT       = Dict(
    "Monitor"         => "Green",
    "Evaluate"        => "Yellow",
    "Assess"          => "Orange",
    "Actionable_Risk" => "Red",
)

# Drought thresholds. SPI is unitless; cutoffs follow McKee (1993) /
# WMO Standardised Precipitation Index User Guide categories.
const CURRENT_SPI3_THRESHOLDS = (above=0.5, normal=-0.5,
                                 mild=-1.0, moderate=-1.5)
const TAIL_SPI_THRESHOLDS     = (nil=-0.5, low=-1.0, moderate=-1.5)
const DEFICIT_THRESHOLDS      = (very_low=0.2, low=0.4, medium=0.6, high=0.8)
const SPATIAL_THRESHOLDS      = (localized=0.3, moderate=0.6)
const TREND_BAND_DEFAULT      = 0.1   # SPI / month

# ============================================================================
# CATEGORISATION
# ============================================================================

"""
Categorise current SPI-3 into 1-5 (drought-side severity, lower index = drier).

  1 Above_Normal      (SPI ≥ +0.5)
  2 Normal            (-0.5 ≤ SPI < +0.5)
  3 Mild_Drought      (-1.0 ≤ SPI < -0.5)
  4 Moderate_Drought  (-1.5 ≤ SPI < -1.0)
  5 Severe_Drought    (SPI < -1.5)

Note: index 5 is "most stressed", matching antecedent_rainfall=Saturated
in the flood model (where flood stress increases with rainfall). The
drought BN uses the same monotonic stress convention.
"""
function categorize_current_spi3(spi::Float64)::Int
    isnan(spi) && return 2  # Normal fallback
    spi <  CURRENT_SPI3_THRESHOLDS.moderate && return 5  # Severe_Drought
    spi <  CURRENT_SPI3_THRESHOLDS.mild     && return 4  # Moderate_Drought
    spi <  CURRENT_SPI3_THRESHOLDS.normal   && return 3  # Mild_Drought
    spi <  CURRENT_SPI3_THRESHOLDS.above    && return 2  # Normal
    return 1  # Above_Normal
end

"""Categorise forecast deficit probability into 1-5 (low → high)."""
function categorize_deficit(p::Float64)::Int
    isnan(p) && return 1
    p < DEFICIT_THRESHOLDS.very_low && return 1  # Very_Low
    p < DEFICIT_THRESHOLDS.low       && return 2  # Low
    p < DEFICIT_THRESHOLDS.medium    && return 3  # Medium
    p < DEFICIT_THRESHOLDS.high      && return 4  # High
    return 5  # Very_High
end

"""Categorise spatial coverage fraction into 1-3."""
function categorize_spatial(coverage::Float64)::Int
    isnan(coverage) && return 1
    coverage < SPATIAL_THRESHOLDS.localized && return 1  # Localized
    coverage < SPATIAL_THRESHOLDS.moderate  && return 2  # Moderate
    return 3  # Widespread
end

"""
Categorise SPI trend into 1-3 from slope (SPI / month). For drought,
"Deteriorating" means slope < -band (SPI dropping). The CSV from
drought_data_prep.py also writes the trend label as a string
(spi3_trend ∈ {Improving, Stable, Deteriorating}); accept either.

Index convention: 1=Improving (least stress), 3=Deteriorating (most stress).
"""
function categorize_spi3_trend(slope::Float64; band::Float64=TREND_BAND_DEFAULT)::Int
    isnan(slope) && return 2  # Stable
    slope >  band && return 1  # Improving
    slope < -band && return 3  # Deteriorating
    return 2  # Stable
end

function categorize_spi3_trend(label::AbstractString)::Int
    s = lowercase(strip(String(label)))
    s == "improving"     && return 1
    s == "deteriorating" && return 3
    return 2  # Stable / Unknown
end

"""Map agreement string to index (1-3)."""
function categorize_agreement(agreement::AbstractString)::Int
    a = lowercase(String(agreement))
    a == "low"  && return 1
    a == "high" && return 3
    return 2
end

"""
Categorise the worst-case ensemble-min SPI (p5 across leads × members) into
tail-risk index (1-4). Lower SPI → higher tail-drought risk.

  1 Nil       (SPI ≥ -0.5)
  2 Low       (-1.0 ≤ SPI < -0.5)
  3 Moderate  (-1.5 ≤ SPI < -1.0)
  4 High      (SPI < -1.5)
"""
function categorize_tail_risk(ens_min_spi::Float64)::Int
    isnan(ens_min_spi) && return 1
    ens_min_spi <  TAIL_SPI_THRESHOLDS.moderate && return 4  # High
    ens_min_spi <  TAIL_SPI_THRESHOLDS.low      && return 3  # Moderate
    ens_min_spi <  TAIL_SPI_THRESHOLDS.nil      && return 2  # Low
    return 1  # Nil
end

# ============================================================================
# CRMA DECISION (cost-loss-ratio rule, identical to flood)
# ============================================================================

function compute_crma_state(risk_probs::Vector{Float64};
                            cost_loss_ratio::Float64=0.2)
    p_minimal  = risk_probs[1]
    p_low      = risk_probs[2]
    p_moderate = risk_probs[3]
    p_high     = risk_probs[4]
    p_extreme  = risk_probs[5]

    p_act      = p_high + p_extreme
    p_assess   = p_moderate + p_high + p_extreme
    p_evaluate = p_low + p_moderate + p_high + p_extreme

    θ_act      = cost_loss_ratio
    θ_assess   = max(2.0 * cost_loss_ratio, 0.40)
    θ_evaluate = max(3.0 * cost_loss_ratio, 0.30)

    if p_act >= θ_act
        expl = "P(High∪Extreme)=$(round(p_act, digits=2)) ≥ C/L=$(round(θ_act, digits=2))"
        return 4, expl
    elseif p_assess >= θ_assess
        expl = "P(Mod∪High∪Extreme)=$(round(p_assess, digits=2)) ≥ $(round(θ_assess, digits=2))"
        return 3, expl
    elseif p_evaluate >= θ_evaluate
        expl = "P(Low∪Mod∪High∪Extreme)=$(round(p_evaluate, digits=2)) ≥ $(round(θ_evaluate, digits=2))"
        return 2, expl
    else
        return 1, "all conditional masses below thresholds"
    end
end

# ============================================================================
# CPT — drought-side compute_risk_probs
# ============================================================================

"""
Compute risk probability vector [5] given parent state indices.

Convention: indices are 1-based with 1 = least drought-stressed,
N = most drought-stressed for each parent. The internal arithmetic
matches flood_bn_ibf_v1.jl's structure but with drought-side weights.

  current_spi3 (1..5)  1=Above_Normal …  5=Severe_Drought
  deficit      (1..5)  1=Very_Low     …  5=Very_High
  spatial      (1..3)  1=Localized    …  3=Widespread
  trend        (1..3)  1=Improving    …  3=Deteriorating
  agreement    (1..3)  1=Low          …  3=High
  tail         (1..4)  1=Nil          …  4=High
"""
function compute_risk_probs(
    current::Int,    # 1-5 (Above_Normal..Severe_Drought)
    deficit::Int,    # 1-5 (Very_Low..Very_High)
    spatial::Int,    # 1-3
    trend::Int,      # 1-3 (Improving..Deteriorating)
    agreement::Int,  # 1-3
    tail::Int=1,     # 1-4 (Nil..High)
)::Vector{Float64}
    c  = current   - 1  # 0..4 (drier ↑)
    d  = deficit   - 1  # 0..4 (more deficit ↑)
    s  = spatial   - 1  # 0..2
    t  = trend     - 1  # 0..2 (worsening ↑)
    ag = agreement - 1
    tr = tail      - 1  # 0..3

    # Base drought-stress score from current state + forecast deficit
    base_risk = c * 0.30 + d * 0.55

    # Spatial modifier (same as flood)
    if s == 2       # Widespread
        base_risk += 0.5
    elseif s == 1   # Moderate
        base_risk += 0.25
    end

    # Trend modifier — Deteriorating SPI raises risk; Improving drops it.
    if t == 2       # Deteriorating
        base_risk += 0.35
    elseif t == 0   # Improving
        base_risk -= 0.30
    end

    # Tail-risk modifier — worst-case ensemble member already deep in
    # drought, even if mean deficit prob is moderate.
    if tr == 3       # High tail (ens_min < -1.5)
        base_risk += 0.60
    elseif tr == 2   # Moderate tail (ens_min < -1.0)
        base_risk += 0.35
    elseif tr == 1   # Low tail (ens_min < -0.5)
        base_risk += 0.10
    end

    # Expert rules — drought analogues of the flood scenario rules.
    probs = if c == 4 && d >= 3 && t == 2
        # Rule 1: Severe_Drought + High deficit + Deteriorating
        if s >= 1
            [0.0, 0.0, 0.05, 0.20, 0.75]
        else
            [0.0, 0.0, 0.10, 0.40, 0.50]
        end
    elseif c >= 3 && d >= 3 && t == 2
        # Rule 2: Moderate/Severe drought + High deficit + Deteriorating
        [0.0, 0.0, 0.10, 0.50, 0.40]
    elseif tr >= 2 && c >= 3 && d <= 2
        # Rule T1: Low mean deficit BUT high tail risk + already dry
        if tr == 3
            [0.0, 0.10, 0.30, 0.45, 0.15]
        else
            [0.05, 0.20, 0.45, 0.25, 0.05]
        end
    elseif tr == 3 && d <= 1
        # Rule T2: Low mean deficit BUT High tail (any current state)
        [0.05, 0.20, 0.40, 0.30, 0.05]
    elseif tr >= 2 && t == 2 && d <= 2
        # Rule T3: Moderate tail + Deteriorating + low mean deficit
        [0.05, 0.15, 0.45, 0.30, 0.05]
    elseif c == 0 && d <= 2 && tr <= 1
        # Rule 3: Above_Normal current + low deficit + no tail risk
        [0.55, 0.35, 0.10, 0.0, 0.0]
    elseif t == 0 && c <= 1 && d <= 1 && tr <= 1
        # Rule 4: Improving + already-good state + low deficit + no tail
        [0.65, 0.30, 0.05, 0.0, 0.0]
    elseif c <= 1 && d >= 3
        # Rule 5: High forecast deficit but currently OK
        [0.10, 0.25, 0.45, 0.15, 0.05]
    elseif base_risk < 1
        [0.50, 0.40, 0.10, 0.0, 0.0]
    elseif base_risk < 2
        [0.10, 0.35, 0.40, 0.15, 0.0]
    elseif base_risk < 3
        [0.05, 0.15, 0.45, 0.30, 0.05]
    elseif base_risk < 4
        [0.0, 0.05, 0.25, 0.50, 0.20]
    else
        [0.0, 0.0, 0.10, 0.40, 0.50]
    end

    # Forecast agreement modifier — Low agreement spreads probability mass.
    uniform = fill(0.20, 5)
    if ag == 0       # Low agreement
        probs = 0.5 .* probs .+ 0.5 .* uniform
    elseif ag == 1   # Medium agreement
        probs = 0.8 .* probs .+ 0.2 .* uniform
    end

    return probs ./ sum(probs)
end

# ============================================================================
# CPT BUILDERS (unchanged shape; mirrors flood)
# ============================================================================

function build_risk_cpt(; include_agreement::Bool=true, include_tail_risk::Bool=false)
    if include_tail_risk
        n_combos = include_agreement ? 5 * 5 * 3 * 3 * 3 * 4 : 5 * 5 * 3 * 3 * 4
    else
        n_combos = include_agreement ? 5 * 5 * 3 * 3 * 3 : 5 * 5 * 3 * 3
    end

    cpt = zeros(Float64, 5, n_combos)
    idx = 0

    if include_tail_risk && include_agreement
        for tl in 1:4, ag in 1:3, tr in 1:3, sp in 1:3, df in 1:5, cu in 1:5
            idx += 1; cpt[:, idx] = compute_risk_probs(cu, df, sp, tr, ag, tl)
        end
    elseif include_tail_risk
        for tl in 1:4, tr in 1:3, sp in 1:3, df in 1:5, cu in 1:5
            idx += 1; cpt[:, idx] = compute_risk_probs(cu, df, sp, tr, 3, tl)
        end
    elseif include_agreement
        for ag in 1:3, tr in 1:3, sp in 1:3, df in 1:5, cu in 1:5
            idx += 1; cpt[:, idx] = compute_risk_probs(cu, df, sp, tr, ag, 1)
        end
    else
        for tr in 1:3, sp in 1:3, df in 1:5, cu in 1:5
            idx += 1; cpt[:, idx] = compute_risk_probs(cu, df, sp, tr, 3, 1)
        end
    end

    return cpt, n_combos
end

"""Tensor form for RxInfer DiscreteTransition."""
function build_risk_cpt_tensor(; include_tail_risk::Bool=true)
    if include_tail_risk
        T = zeros(Float64, 5, 5, 5, 3, 3, 4)
        for tl in 1:4, tr in 1:3, sp in 1:3, df in 1:5, cu in 1:5
            T[:, cu, df, sp, tr, tl] = compute_risk_probs(cu, df, sp, tr, 3, tl)
        end
        return T
    else
        T = zeros(Float64, 5, 5, 5, 3, 3)
        for tr in 1:3, sp in 1:3, df in 1:5, cu in 1:5
            T[:, cu, df, sp, tr] = compute_risk_probs(cu, df, sp, tr, 3, 1)
        end
        return T
    end
end

function build_action_cpt()::Matrix{Float64}
    return [
        0.95  0.15  0.00  0.00  0.00;  # Monitor
        0.05  0.80  0.20  0.05  0.00;  # Alert
        0.00  0.05  0.75  0.25  0.05;  # Prepare
        0.00  0.00  0.05  0.70  0.95;  # Act
    ]
end

# ============================================================================
# RxInfer MODEL
# ============================================================================

@model function drought_bn_model_5parent(T, cur_data, def_data, spa_data, trn_data, tail_data, risk_data)
    cur  ~ Categorical(fill(1/5, 5))
    def  ~ Categorical(fill(1/5, 5))
    spa  ~ Categorical(fill(1/3, 3))
    trn  ~ Categorical(fill(1/3, 3))
    tail ~ Categorical(fill(1/4, 4))
    cur_data  ~ DiscreteTransition(cur,  diageye(5))
    def_data  ~ DiscreteTransition(def,  diageye(5))
    spa_data  ~ DiscreteTransition(spa,  diageye(3))
    trn_data  ~ DiscreteTransition(trn,  diageye(3))
    tail_data ~ DiscreteTransition(tail, diageye(4))
    risk ~ DiscreteTransition(cur, T, def, spa, trn, tail)
    risk_data ~ DiscreteTransition(risk, diageye(5))
end

@model function drought_bn_model_4parent(T, cur_data, def_data, spa_data, trn_data, risk_data)
    cur  ~ Categorical(fill(1/5, 5))
    def  ~ Categorical(fill(1/5, 5))
    spa  ~ Categorical(fill(1/3, 3))
    trn  ~ Categorical(fill(1/3, 3))
    cur_data  ~ DiscreteTransition(cur,  diageye(5))
    def_data  ~ DiscreteTransition(def,  diageye(5))
    spa_data  ~ DiscreteTransition(spa,  diageye(3))
    trn_data  ~ DiscreteTransition(trn,  diageye(3))
    risk ~ DiscreteTransition(cur, T, def, spa, trn)
    risk_data ~ DiscreteTransition(risk, diageye(5))
end

_rxinfer_init_5 = @initialization begin
    q(cur)  = Categorical(fill(1/5, 5))
    q(def)  = Categorical(fill(1/5, 5))
    q(spa)  = Categorical(fill(1/3, 3))
    q(trn)  = Categorical(fill(1/3, 3))
    q(tail) = Categorical(fill(1/4, 4))
    q(risk) = Categorical(fill(1/5, 5))
end

_rxinfer_init_4 = @initialization begin
    q(cur)  = Categorical(fill(1/5, 5))
    q(def)  = Categorical(fill(1/5, 5))
    q(spa)  = Categorical(fill(1/3, 3))
    q(trn)  = Categorical(fill(1/3, 3))
    q(risk) = Categorical(fill(1/5, 5))
end

function infer_rxinfer_soft(
    cur_ev::Vector{Float64},
    def_ev::Vector{Float64},
    spa_ev::Vector{Float64},
    trn_ev::Vector{Float64};
    tail_ev::Union{Nothing,Vector{Float64}}=nothing,
    risk_cpt_tensor::AbstractArray{Float64},
    action_cpt::Matrix{Float64},
    iterations::Int=10,
)::Tuple{Vector{Float64},Vector{Float64}}
    if tail_ev === nothing
        r = infer(
            model = drought_bn_model_4parent(T = risk_cpt_tensor),
            data  = (cur_data = cur_ev, def_data = def_ev, spa_data = spa_ev,
                     trn_data = trn_ev, risk_data = missing),
            iterations     = iterations,
            initialization = _rxinfer_init_4,
        )
    else
        r = infer(
            model = drought_bn_model_5parent(T = risk_cpt_tensor),
            data  = (cur_data = cur_ev, def_data = def_ev, spa_data = spa_ev,
                     trn_data = trn_ev, tail_data = tail_ev, risk_data = missing),
            iterations     = iterations,
            initialization = _rxinfer_init_5,
        )
    end
    risk_probs   = Vector{Float64}(last(r.posteriors[:risk]).p)
    action_probs = action_cpt * risk_probs
    return risk_probs, action_probs
end

# ============================================================================
# DIRECT MATMUL FALLBACK
# ============================================================================

function encode_parents(cur::Int, def::Int, spa::Int, trn::Int, agr::Int;
                        tail::Int=1, include_tail_risk::Bool=false)::Int
    if include_tail_risk
        return ((tail - 1) * 3 * 3 * 3 * 5 * 5 +
                (agr  - 1) * 3 * 3 * 5 * 5 +
                (trn  - 1) * 3 * 5 * 5 +
                (spa  - 1) * 5 * 5 +
                (def  - 1) * 5 +
                (cur  - 1)) + 1
    else
        return ((agr  - 1) * 3 * 3 * 5 * 5 +
                (trn  - 1) * 3 * 5 * 5 +
                (spa  - 1) * 5 * 5 +
                (def  - 1) * 5 +
                (cur  - 1)) + 1
    end
end

function encode_parents_no_agreement(cur::Int, def::Int, spa::Int, trn::Int;
                                     tail::Int=1, include_tail_risk::Bool=false)::Int
    if include_tail_risk
        return ((tail - 1) * 3 * 3 * 5 * 5 +
                (trn  - 1) * 3 * 5 * 5 +
                (spa  - 1) * 5 * 5 +
                (def  - 1) * 5 +
                (cur  - 1)) + 1
    else
        return ((trn  - 1) * 3 * 5 * 5 +
                (spa  - 1) * 5 * 5 +
                (def  - 1) * 5 +
                (cur  - 1)) + 1
    end
end

function infer_direct(
    cur_idx::Int, def_idx::Int, spa_idx::Int, trn_idx::Int, agr_idx::Int,
    risk_cpt::Matrix{Float64}, action_cpt::Matrix{Float64};
    include_agreement::Bool=true,
    tail_risk_idx::Int=1, include_tail_risk::Bool=false,
)
    parent_idx = if include_agreement
        encode_parents(cur_idx, def_idx, spa_idx, trn_idx, agr_idx;
                        tail=tail_risk_idx, include_tail_risk)
    else
        encode_parents_no_agreement(cur_idx, def_idx, spa_idx, trn_idx;
                                     tail=tail_risk_idx, include_tail_risk)
    end
    risk_probs = risk_cpt[:, parent_idx]
    action_probs = action_cpt * risk_probs
    return risk_probs, action_probs
end

"""
Tensor-contraction soft-evidence inference. Mathematically equivalent to
RxInfer message passing but ~1000× faster — used for bulk per-member runs.
"""
function infer_soft_matmul(
    cur_ev::Vector{Float64}, def_ev::Vector{Float64},
    spa_ev::Vector{Float64}, trn_ev::Vector{Float64},
    tail_ev::Vector{Float64},
    T::Array{Float64},
    action_cpt::Matrix{Float64},
)::Tuple{Vector{Float64},Vector{Float64}}
    risk_probs = zeros(Float64, 5)
    @inbounds for tl in 1:4, tr in 1:3, sp in 1:3, df in 1:5, cu in 1:5
        w = cur_ev[cu] * def_ev[df] * spa_ev[sp] * trn_ev[tr] * tail_ev[tl]
        for r in 1:5
            risk_probs[r] += T[r, cu, df, sp, tr, tl] * w
        end
    end
    s = sum(risk_probs)
    if s > 0; risk_probs ./= s; end
    return risk_probs, action_cpt * risk_probs
end

onehot(idx::Int, k::Int) = (v = zeros(Float64, k); v[idx] = 1.0; v)

# ============================================================================
# BOUNDARY PROCESSING
# ============================================================================

struct BoundaryInput
    id::String
    name::String
    country::String
    current_spi3::Float64
    spi3_trend::String
    forecast_deficit_prob::Float64
    spatial_coverage::Float64
    forecast_agreement::String
    ens_min_spi::Float64
    cur_probs::Union{Nothing,Vector{Float64}}
    def_probs::Union{Nothing,Vector{Float64}}
    spa_probs::Union{Nothing,Vector{Float64}}
    trn_probs::Union{Nothing,Vector{Float64}}
    tail_probs::Union{Nothing,Vector{Float64}}
end

BoundaryInput(id, name, country, cur_spi, trend, def_p, spa_cov, agr, ens_min) =
    BoundaryInput(id, name, country, cur_spi, trend, def_p, spa_cov, agr, ens_min,
                  nothing, nothing, nothing, nothing, nothing)

struct BoundaryResult
    boundary_id::String
    boundary_name::String
    country::String
    current_spi3_category::String
    spi3_trend::String
    risk_level::String
    risk_probabilities::Vector{Float64}
    recommended_action::String         # deprecated
    action_probabilities::Vector{Float64}
    confidence::Float64
    crma_state::String
    crma_explanation::String
    traffic_light::String
end

function _assemble_result(b::BoundaryInput, cur_idx::Int, trn_idx::Int,
                          risk_probs::Vector{Float64}, action_probs::Vector{Float64},
                          cost_loss_ratio::Float64)::BoundaryResult
    crma_idx, crma_expl = compute_crma_state(risk_probs; cost_loss_ratio)
    crma_state = CRMA_STATES[crma_idx]
    return BoundaryResult(
        b.id, b.name, b.country,
        CURRENT_SPI3_STATES[cur_idx],
        TREND_STATES[trn_idx],
        RISK_STATES[argmax(risk_probs)],
        risk_probs,
        ACTION_STATES[argmax(action_probs)],
        action_probs,
        maximum(action_probs),
        crma_state, crma_expl, TRAFFIC_LIGHT[crma_state],
    )
end

function process_boundary(
    b::BoundaryInput, risk_cpt::Matrix{Float64}, action_cpt::Matrix{Float64};
    include_agreement::Bool=true, include_tail_risk::Bool=false,
    cost_loss_ratio::Float64=0.2,
)::BoundaryResult
    cur_idx = categorize_current_spi3(b.current_spi3)
    def_idx = categorize_deficit(b.forecast_deficit_prob)
    spa_idx = categorize_spatial(b.spatial_coverage)
    trn_idx = categorize_spi3_trend(b.spi3_trend)
    agr_idx = categorize_agreement(b.forecast_agreement)
    tl_idx  = categorize_tail_risk(b.ens_min_spi)

    risk_probs, action_probs = infer_direct(
        cur_idx, def_idx, spa_idx, trn_idx, agr_idx,
        risk_cpt, action_cpt;
        include_agreement, tail_risk_idx=tl_idx, include_tail_risk,
    )

    return _assemble_result(b, cur_idx, trn_idx, risk_probs, action_probs, cost_loss_ratio)
end

function process_boundary_rxinfer(
    b::BoundaryInput, risk_cpt_tensor::AbstractArray{Float64},
    action_cpt::Matrix{Float64};
    include_tail_risk::Bool=false,
    cost_loss_ratio::Float64=0.2, iterations::Int=10,
)::BoundaryResult
    cur_idx = categorize_current_spi3(b.current_spi3)
    def_idx = categorize_deficit(b.forecast_deficit_prob)
    spa_idx = categorize_spatial(b.spatial_coverage)
    trn_idx = categorize_spi3_trend(b.spi3_trend)
    tl_idx  = categorize_tail_risk(b.ens_min_spi)

    cur_ev = b.cur_probs === nothing ? onehot(cur_idx, 5) : b.cur_probs
    def_ev = b.def_probs === nothing ? onehot(def_idx, 5) : b.def_probs
    spa_ev = b.spa_probs === nothing ? onehot(spa_idx, 3) : b.spa_probs
    trn_ev = b.trn_probs === nothing ? onehot(trn_idx, 3) : b.trn_probs
    tail_ev = include_tail_risk ?
              (b.tail_probs === nothing ? onehot(tl_idx, 4) : b.tail_probs) :
              nothing

    risk_probs, action_probs = infer_rxinfer_soft(
        cur_ev, def_ev, spa_ev, trn_ev;
        tail_ev = tail_ev,
        risk_cpt_tensor = risk_cpt_tensor,
        action_cpt = action_cpt,
        iterations = iterations,
    )
    return _assemble_result(b, cur_idx, trn_idx, risk_probs, action_probs, cost_loss_ratio)
end

function process_all_boundaries(
    boundaries::Vector{BoundaryInput};
    include_agreement::Bool=true, include_tail_risk::Bool=false,
    cost_loss_ratio::Float64=0.2, use_rxinfer::Bool=true,
)::Vector{BoundaryResult}
    action_cpt = build_action_cpt()
    results = Vector{BoundaryResult}(undef, length(boundaries))

    if use_rxinfer && !include_agreement
        T = build_risk_cpt_tensor(; include_tail_risk)
        for (i, b) in enumerate(boundaries)
            results[i] = process_boundary_rxinfer(b, T, action_cpt;
                                                  include_tail_risk, cost_loss_ratio)
            i % 50 == 0 && @info "Processed $i/$(length(boundaries)) boundaries (RxInfer)"
        end
    else
        if use_rxinfer && include_agreement
            @info "include_agreement=true has 6 parents; falling back to matmul"
        end
        risk_cpt, _ = build_risk_cpt(; include_agreement, include_tail_risk)
        for (i, b) in enumerate(boundaries)
            results[i] = process_boundary(b, risk_cpt, action_cpt;
                                          include_agreement, include_tail_risk, cost_loss_ratio)
            i % 50 == 0 && @info "Processed $i/$(length(boundaries)) boundaries (matmul)"
        end
    end
    return results
end

# ============================================================================
# DBN — month-to-month temporal coupling
# ============================================================================

function blend_temporal_prior(yesterday::Vector{Float64}; decay::Float64=0.6)::Vector{Float64}
    v = decay .* yesterday .+ (1.0 - decay) .* fill(0.2, 5)
    return v ./ sum(v)
end

"""
Run the BN as a DBN across a sequence of monthly input CSVs. Last month's
risk posterior is blended into this month as soft virtual evidence.
After `lookback` consecutive months (default 6 = forecast horizon) the
chain resets to a uniform prior.
"""
function run_dbn_sequence(
    input_csvs::Vector{String};
    include_tail_risk::Bool=true, cost_loss_ratio::Float64=0.2,
    temporal_decay::Float64=0.6, lookback::Int=6,
)
    T = build_risk_cpt_tensor(; include_tail_risk)
    action_cpt = build_action_cpt()
    prev = Dict{String, Vector{Float64}}()
    chain_len = Dict{String, Int}()
    all_frames = DataFrames.DataFrame[]

    for (month_idx, csv_path) in enumerate(input_csvs)
        df = CSV.read(csv_path, DataFrames.DataFrame)
        colnames = names(df)
        has_min = "ens_min_spi" in colnames

        _soft(prefix, k, row) = all("$(prefix)_p$i" in colnames for i in 1:k) ?
            Float64[row["$(prefix)_p$i"] for i in 1:k] : nothing

        target_date = "target_date" in colnames ? string(df[1, :target_date]) : "month_$month_idx"
        n = DataFrames.nrow(df)
        out_rows = Vector{NamedTuple}(undef, n)

        for (i, row) in enumerate(DataFrames.eachrow(df))
            bid = String(row.id)
            cur_idx = categorize_current_spi3(Float64(row.current_spi3))
            def_idx = categorize_deficit(Float64(row.forecast_deficit_prob))
            spa_idx = categorize_spatial(Float64(row.spatial_coverage))
            trn_idx = categorize_spi3_trend(String(row.spi3_trend))
            tl_idx  = has_min ? categorize_tail_risk(Float64(row.ens_min_spi)) : 1

            cur_ev  = something(_soft("cur",  5, row), onehot(cur_idx, 5))
            def_ev  = something(_soft("def",  5, row), onehot(def_idx, 5))
            spa_ev  = something(_soft("spa",  3, row), onehot(spa_idx, 3))
            trn_ev  = something(_soft("trn",  3, row), onehot(trn_idx, 3))
            tail_ev = include_tail_risk ?
                      something(_soft("tail", 4, row), onehot(tl_idx, 4)) :
                      onehot(1, 4)

            yesterday = get(prev, bid, nothing)
            cl = get(chain_len, bid, 0)
            risk_ev = (yesterday !== nothing && cl < lookback) ?
                      blend_temporal_prior(yesterday; decay=temporal_decay) : nothing

            risk_probs, action_probs = infer_soft_matmul(
                cur_ev, def_ev, spa_ev, trn_ev, tail_ev, T, action_cpt)

            if risk_ev !== nothing
                risk_probs .*= risk_ev
                s = sum(risk_probs)
                if s > 0; risk_probs ./= s; end
                action_probs = action_cpt * risk_probs
            end

            prev[bid] = copy(risk_probs)
            chain_len[bid] = (yesterday !== nothing ? cl + 1 : 1)
            crma_idx, crma_expl = compute_crma_state(risk_probs; cost_loss_ratio)

            out_rows[i] = (
                target_date     = target_date,
                dbn_month       = month_idx,
                boundary_id     = bid,
                boundary_name   = String(row.name),
                country         = String(row.country),
                risk_level      = RISK_STATES[argmax(risk_probs)],
                crma_state      = CRMA_STATES[crma_idx],
                traffic_light   = TRAFFIC_LIGHT[CRMA_STATES[crma_idx]],
                crma_explanation = crma_expl,
                risk_minimal    = risk_probs[1],
                risk_low        = risk_probs[2],
                risk_moderate   = risk_probs[3],
                risk_high       = risk_probs[4],
                risk_extreme    = risk_probs[5],
                temporal_prior  = risk_ev !== nothing,
                p_high_extreme  = risk_probs[4] + risk_probs[5],
            )
        end
        push!(all_frames, DataFrames.DataFrame(out_rows))
        @info "DBN month $month_idx ($target_date): $n boundaries"
    end
    return vcat(all_frames...)
end

# ============================================================================
# PER-MEMBER STORYLINES
# ============================================================================

function run_per_member_bn(
    member_csv::String;
    include_tail_risk::Bool=true, cost_loss_ratio::Float64=0.2,
)
    df = CSV.read(member_csv, DataFrames.DataFrame)
    T = build_risk_cpt_tensor(; include_tail_risk)
    action_cpt = build_action_cpt()
    colnames = names(df)
    _soft(prefix, k, row) = all("$(prefix)_p$i" in colnames for i in 1:k) ?
        Float64[row["$(prefix)_p$i"] for i in 1:k] : nothing

    n = DataFrames.nrow(df)
    out = Vector{NamedTuple}(undef, n)

    for (i, row) in enumerate(DataFrames.eachrow(df))
        cur_idx = categorize_current_spi3(Float64(row.current_spi3))
        def_idx = categorize_deficit(Float64(row.member_def_frac))
        spa_idx = categorize_spatial(Float64(row.member_spa_cov))
        trn_idx = categorize_spi3_trend(String(row.spi3_trend))
        tl_idx  = categorize_tail_risk(Float64(row.member_min_spi))

        cur_ev  = something(_soft("cur",  5, row), onehot(cur_idx, 5))
        def_ev  = something(_soft("def",  5, row), onehot(def_idx, 5))
        spa_ev  = something(_soft("spa",  3, row), onehot(spa_idx, 3))
        trn_ev  = something(_soft("trn",  3, row), onehot(trn_idx, 3))
        tail_ev = something(_soft("tail", 4, row), onehot(tl_idx, 4))

        risk_probs, _ = infer_soft_matmul(
            cur_ev, def_ev, spa_ev, trn_ev, tail_ev, T, action_cpt)
        crma_idx, _ = compute_crma_state(risk_probs; cost_loss_ratio)

        out[i] = (
            boundary_id     = String(row.boundary_id),
            boundary_name   = String(row.boundary_name),
            country         = String(row.country),
            member          = String(row.member),
            target_date     = string(row.target_date),
            risk_level      = RISK_STATES[argmax(risk_probs)],
            crma_state      = CRMA_STATES[crma_idx],
            p_high_extreme  = risk_probs[4] + risk_probs[5],
            risk_minimal    = risk_probs[1],
            risk_low        = risk_probs[2],
            risk_moderate   = risk_probs[3],
            risk_high       = risk_probs[4],
            risk_extreme    = risk_probs[5],
            member_min_spi   = Float64(row.member_min_spi),
            member_def_frac  = Float64(row.member_def_frac),
        )
    end
    return DataFrames.DataFrame(out)
end

function select_storylines(member_results::DataFrames.DataFrame)
    groups = DataFrames.groupby(member_results, [:boundary_id, :target_date])
    rows = NamedTuple[]
    for g in groups
        sorted = sort(g, :p_high_extreme, rev=true)
        n = DataFrames.nrow(sorted)
        picks = [
            ("worst",  sorted[1, :]),
            ("median", sorted[div(n, 2) + 1, :]),
            ("best",   sorted[n, :]),
        ]
        for (stype, r) in picks
            n_ge = sum(sorted.p_high_extreme .>= r.p_high_extreme)
            push!(rows, (
                storyline       = stype,
                boundary_id     = r.boundary_id,
                boundary_name   = r.boundary_name,
                country         = r.country,
                target_date     = r.target_date,
                member          = r.member,
                risk_level      = r.risk_level,
                crma_state      = r.crma_state,
                p_high_extreme  = r.p_high_extreme,
                risk_minimal    = r.risk_minimal,
                risk_low        = r.risk_low,
                risk_moderate   = r.risk_moderate,
                risk_high       = r.risk_high,
                risk_extreme    = r.risk_extreme,
                member_min_spi   = r.member_min_spi,
                probability     = round(n_ge / n, digits=3),
                n_members       = n,
            ))
        end
    end
    return DataFrames.DataFrame(rows)
end

# ============================================================================
# CSV DRIVER
# ============================================================================

function run_csv(input_csv::String, output_csv::String;
                 include_agreement::Bool, include_tail_risk::Bool,
                 cost_loss_ratio::Float64=0.2, use_rxinfer::Bool=true)
    df = CSV.read(input_csv, DataFrames.DataFrame)
    colnames = names(df)
    has_min = "ens_min_spi" in colnames
    if include_tail_risk && !has_min
        @warn "--tail-risk requested but ens_min_spi column missing; disabling"
        include_tail_risk = false
    end

    # Tolerate missing/empty cells from the prep CSV (e.g. boundaries with
    # no obs pixel hits): replace with NaN / sentinels so the categorise_*
    # functions take their NaN branch.
    _f(x) = (x === missing || x === nothing) ? NaN : Float64(x)
    _s(x) = (x === missing || x === nothing) ? "" : String(x)
    _soft(prefix::String, k::Int, row) = all("$(prefix)_p$i" in colnames for i in 1:k) ?
        [_f(row["$(prefix)_p$i"]) for i in 1:k] : nothing

    inputs = Vector{BoundaryInput}(undef, DataFrames.nrow(df))
    n_soft_rows = 0
    for (i, row) in enumerate(DataFrames.eachrow(df))
        cur_p  = _soft("cur",  5, row)
        def_p  = _soft("def",  5, row)
        spa_p  = _soft("spa",  3, row)
        trn_p  = _soft("trn",  3, row)
        tail_p = _soft("tail", 4, row)
        if any(x -> x !== nothing, (cur_p, def_p, spa_p, trn_p, tail_p))
            n_soft_rows += 1
        end
        inputs[i] = BoundaryInput(
            _s(row.id),
            _s(row.name),
            _s(row.country),
            _f(row.current_spi3),
            _s(row.spi3_trend),
            _f(row.forecast_deficit_prob),
            _f(row.spatial_coverage),
            _s(row.forecast_agreement),
            has_min ? _f(row.ens_min_spi) : 0.0,
            cur_p, def_p, spa_p, trn_p, tail_p,
        )
    end

    backend = use_rxinfer && !include_agreement ? "RxInfer" : "matmul"
    @info "Processing $(length(inputs)) boundaries (backend=$backend agreement=$include_agreement tail_risk=$include_tail_risk C/L=$cost_loss_ratio soft_rows=$n_soft_rows)"
    results = process_all_boundaries(inputs; include_agreement, include_tail_risk,
                                      cost_loss_ratio, use_rxinfer)

    out = DataFrames.DataFrame(
        boundary_id           = [r.boundary_id for r in results],
        boundary_name         = [r.boundary_name for r in results],
        country               = [r.country for r in results],
        current_spi3_category = [r.current_spi3_category for r in results],
        spi3_trend            = [r.spi3_trend for r in results],
        risk_level            = [r.risk_level for r in results],
        crma_state            = [r.crma_state for r in results],
        traffic_light         = [r.traffic_light for r in results],
        crma_explanation      = [r.crma_explanation for r in results],
        recommended_action    = [r.recommended_action for r in results],
        confidence            = [r.confidence for r in results],
        risk_minimal          = [r.risk_probabilities[1] for r in results],
        risk_low              = [r.risk_probabilities[2] for r in results],
        risk_moderate         = [r.risk_probabilities[3] for r in results],
        risk_high             = [r.risk_probabilities[4] for r in results],
        risk_extreme          = [r.risk_probabilities[5] for r in results],
        action_monitor        = [r.action_probabilities[1] for r in results],
        action_alert          = [r.action_probabilities[2] for r in results],
        action_prepare        = [r.action_probabilities[3] for r in results],
        action_act            = [r.action_probabilities[4] for r in results],
    )

    mkpath(dirname(abspath(output_csv)))
    CSV.write(output_csv, out)
    @info "Wrote $output_csv rows=$(DataFrames.nrow(out))"

    risk_counts = DataFrames.combine(DataFrames.groupby(out, :risk_level), DataFrames.nrow => :n)
    @info "Risk distribution:" risk_counts
    crma_counts = DataFrames.combine(DataFrames.groupby(out, :crma_state), DataFrames.nrow => :n)
    @info "CRMA state distribution:" crma_counts
end

# ============================================================================
# CLI
# ============================================================================

function getarg(flag::String)
    i = findfirst(==(flag), ARGS)
    return i === nothing || i == length(ARGS) ? nothing : ARGS[i + 1]
end

function self_test()
    @info "Running drought BN self-test..."
    risk_cpt, _ = build_risk_cpt(; include_agreement=true)
    action_cpt  = build_action_cpt()

    # 1. Worst case: Severe_Drought + Very_High deficit + Widespread + Deteriorating + High agreement
    rp, ap = infer_direct(5, 5, 3, 3, 3, risk_cpt, action_cpt)
    @info "Test 1 (worst case):" risk=RISK_STATES[argmax(rp)] action=ACTION_STATES[argmax(ap)]
    @assert RISK_STATES[argmax(rp)] == "Extreme" "Expected Extreme risk"
    @assert ACTION_STATES[argmax(ap)] == "Act" "Expected Act"

    # 2. Best case: Above_Normal + Very_Low + Localized + Improving + High agreement
    rp2, ap2 = infer_direct(1, 1, 1, 1, 3, risk_cpt, action_cpt)
    @info "Test 2 (best case):" risk=RISK_STATES[argmax(rp2)] action=ACTION_STATES[argmax(ap2)]
    @assert RISK_STATES[argmax(rp2)] == "Minimal" "Expected Minimal"
    @assert ACTION_STATES[argmax(ap2)] == "Monitor" "Expected Monitor"

    # 3. Low agreement spreads probabilities
    rp_high, _ = infer_direct(3, 3, 2, 2, 3, risk_cpt, action_cpt)
    rp_low,  _ = infer_direct(3, 3, 2, 2, 1, risk_cpt, action_cpt)
    H_high = -sum(p * log(max(p, 1e-10)) for p in rp_high)
    H_low  = -sum(p * log(max(p, 1e-10)) for p in rp_low)
    @assert H_low > H_high "Low agreement should increase entropy"
    @info "Test 3 (agreement entropy):" H_low H_high
    @info "All self-tests passed."
end

function main()
    if "--test" in ARGS
        self_test()
        return
    end

    input_csv  = getarg("--input-csv")
    output_csv = getarg("--output-csv")
    include_agreement = !("--no-agreement" in ARGS)
    include_tail_risk = "--tail-risk" in ARGS
    cl_str = getarg("--cost-loss-ratio")
    cost_loss_ratio = cl_str === nothing ? 0.2 : parse(Float64, cl_str)
    use_rxinfer = !("--legacy-inference" in ARGS)

    if input_csv !== nothing && output_csv !== nothing
        run_csv(input_csv, output_csv; include_agreement, include_tail_risk,
                cost_loss_ratio, use_rxinfer)
        return
    end

    @info "Drought BN IBF v1 (Julia/RxInfer)"
    @info "Usage: julia drought_bn_ibf_v1.jl --input-csv IN.csv --output-csv OUT.csv [--no-agreement] [--tail-risk] [--legacy-inference] [--cost-loss-ratio 0.2]"
    @info "       julia drought_bn_ibf_v1.jl --test"

    # Demo: a moderately-stressed boundary
    b = BoundaryInput(
        "ETH.1", "Tigray", "Ethiopia",
        -1.4, "Deteriorating",        # current SPI-3, trend
        0.55, 0.45, "Medium", -1.7,   # deficit prob, spatial, agreement, ens-min SPI
    )
    risk_cpt, _ = build_risk_cpt()
    action_cpt = build_action_cpt()
    result = process_boundary(b, risk_cpt, action_cpt; include_tail_risk=true)
    @info "Demo result:" boundary=result.boundary_id risk=result.risk_level crma=result.crma_state confidence=@sprintf("%.2f", result.confidence)

    println("\nRisk probabilities:")
    for (state, prob) in zip(RISK_STATES, result.risk_probabilities)
        bar = repeat("█", round(Int, prob * 40))
        @printf("  %-10s %5.1f%% %s\n", state, prob * 100, bar)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
