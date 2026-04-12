#=
Flood Impact-Based Forecasting using Bayesian Networks - RxInfer.jl Port

Port of flood_bn_ibf_v1.py Bayesian Network to Julia/RxInfer.jl
Uses reactive message passing for inference instead of variable elimination.

Dependencies (add to Project.toml or install manually):
    using Pkg
    Pkg.add(["RxInfer", "CSV", "DataFrames", "JSON3"])

Usage:
    julia --project flood_bn_ibf_v1.jl --boundaries path/to/boundaries.geojson --date 2026-01-15

Author: ICPAC IBF Team
Date: April 2026
=#

using LinearAlgebra
using Printf
using CSV
using DataFrames

# RxInfer is optional: only needed for the reactive message-passing demo path.
# Direct inference (infer_direct) uses pure matrix math and works without it.
const HAS_RXINFER = try
    @eval using RxInfer
    true
catch
    false
end

# ============================================================================
# CONSTANTS
# ============================================================================

# State labels for each node
const ANTECEDENT_STATES = ["Dry", "Normal", "Wet", "Very_Wet", "Saturated"]  # 5
const EXCEEDANCE_STATES = ["Very_Low", "Low", "Medium", "High", "Very_High"]  # 5
const SPATIAL_STATES    = ["Localized", "Moderate", "Widespread"]             # 3
const TREND_STATES      = ["Decreasing", "Stable", "Increasing"]             # 3
const AGREEMENT_STATES  = ["Low", "Medium", "High"]                          # 3
const TAIL_RISK_STATES  = ["None", "Low", "Moderate", "High"]                # 4
const RISK_STATES       = ["Minimal", "Low", "Moderate", "High", "Extreme"]  # 5
const ACTION_STATES     = ["Monitor", "Alert", "Prepare", "Act"]             # 4

# Precipitation thresholds (mm/24h)
const PRECIP_THRESHOLDS_24H = Dict(
    "light"       => 5,
    "moderate"    => 25,
    "heavy"       => 50,
    "very_heavy"  => 75,
    "extreme"     => 100,
    "exceptional" => 125,
)

# Antecedent rainfall thresholds (7-day accumulated, mm)
const ANTECEDENT_THRESHOLDS = Dict(
    "dry"       => 10.0,
    "normal"    => 30.0,
    "wet"       => 60.0,
    "very_wet"  => 100.0,
    "saturated" => Inf,
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
Categorize 7-day accumulated rainfall into antecedent condition index (1-5).
"""
function categorize_antecedent(rainfall_mm::Float64)::Int
    isnan(rainfall_mm) && return 2  # Normal
    rainfall_mm < ANTECEDENT_THRESHOLDS["dry"]      && return 1  # Dry
    rainfall_mm < ANTECEDENT_THRESHOLDS["normal"]    && return 2  # Normal
    rainfall_mm < ANTECEDENT_THRESHOLDS["wet"]       && return 3  # Wet
    rainfall_mm < ANTECEDENT_THRESHOLDS["very_wet"]  && return 4  # Very_Wet
    return 5  # Saturated
end

"""
Categorize exceedance probability (0-1) into discrete state index (1-5).
"""
function categorize_exceedance(eprob::Float64)::Int
    isnan(eprob) && return 1
    eprob < 0.2 && return 1  # Very_Low
    eprob < 0.4 && return 2  # Low
    eprob < 0.6 && return 3  # Medium
    eprob < 0.8 && return 4  # High
    return 5  # Very_High
end

"""
Categorize spatial coverage fraction (0-1) into discrete state index (1-3).
"""
function categorize_spatial(coverage::Float64)::Int
    isnan(coverage) && return 1
    coverage < 0.3 && return 1  # Localized
    coverage < 0.6 && return 2  # Moderate
    return 3  # Widespread
end

"""
Map trend string to index (1-3).
"""
function categorize_trend(trend::String)::Int
    t = lowercase(trend)
    t == "decreasing" && return 1
    t == "increasing" && return 3
    return 2  # Stable
end

"""
Map agreement string to index (1-3).
"""
function categorize_agreement(agreement::String)::Int
    a = lowercase(agreement)
    a == "low"  && return 1
    a == "high" && return 3
    return 2  # Medium
end

"""
Categorize ensemble-max / threshold ratio into tail-risk index (1-4).
Captures whether any single ensemble member exceeds the RP threshold.
"""
function categorize_tail_risk(max_ratio::Float64)::Int
    isnan(max_ratio) && return 1
    max_ratio < 0.5 && return 1  # None — well below threshold
    max_ratio < 1.0 && return 2  # Low — approaching threshold
    max_ratio < 2.0 && return 3  # Moderate — at least 1 member exceeds
    return 4  # High — member well above threshold (≥ 2× RP)
end

# ============================================================================
# CPT CONSTRUCTION (mirrors _compute_risk_probs from Python)
# ============================================================================

"""
Compute risk probability vector [5] given parent state indices.
Extends the original expert rules with a tail_risk node that captures
whether any single ensemble member exceeds the RP threshold.
"""
function compute_risk_probs(
    antecedent::Int,  # 1-5
    exceed::Int,      # 1-5
    spatial::Int,     # 1-3
    trend::Int,       # 1-3
    agreement::Int,   # 1-3
    tail::Int=1,      # 1-4 (None, Low, Moderate, High)
)::Vector{Float64}
    # Convert to 0-based for the arithmetic (matching Python)
    a = antecedent - 1
    e = exceed - 1
    s = spatial - 1
    t = trend - 1
    ag = agreement - 1
    tr = tail - 1

    # Base risk score
    base_risk = a * 0.30 + e * 0.55

    # Spatial modifier
    if s == 2       # Widespread
        base_risk += 0.5
    elseif s == 1   # Moderate
        base_risk += 0.25
    end

    # Trend modifier
    if t == 2       # Increasing
        base_risk += 0.35
    elseif t == 0   # Decreasing
        base_risk -= 0.30
    end

    # Tail risk modifier: boost risk when ensemble max exceeds threshold
    # even if mean exceedance probability is low
    if tr == 3       # High: ens_max ≥ 2× threshold
        base_risk += 0.60
    elseif tr == 2   # Moderate: ens_max exceeds threshold (1-2×)
        base_risk += 0.35
    elseif tr == 1   # Low: approaching threshold (0.5-1.0×)
        base_risk += 0.10
    end

    # Expert rules for specific scenarios
    probs = if a == 4 && e >= 3 && t == 2
        # Rule 1: Saturated + High/Very_High + Increasing
        if s >= 1
            [0.0, 0.0, 0.05, 0.20, 0.75]
        else
            [0.0, 0.0, 0.10, 0.40, 0.50]
        end
    elseif a >= 3 && e >= 3 && t == 2
        # Rule 2: Very_Wet/Saturated + High + Increasing
        [0.0, 0.0, 0.10, 0.50, 0.40]
    # Rule T1: Low mean exceedance BUT high tail risk + wet/saturated ground
    elseif tr >= 2 && a >= 3 && e <= 2
        if tr == 3  # High tail risk
            [0.0, 0.10, 0.30, 0.45, 0.15]
        else        # Moderate tail risk
            [0.05, 0.20, 0.45, 0.25, 0.05]
        end
    # Rule T2: Low mean exceedance BUT high tail risk (any antecedent)
    elseif tr == 3 && e <= 1
        [0.05, 0.20, 0.40, 0.30, 0.05]
    # Rule T3: Moderate tail risk + increasing trend
    elseif tr >= 2 && t == 2 && e <= 2
        [0.05, 0.15, 0.45, 0.30, 0.05]
    elseif a == 0 && e <= 2 && tr <= 1
        # Rule 3: Dry + low forecast + no tail risk
        [0.55, 0.35, 0.10, 0.0, 0.0]
    elseif t == 0 && a <= 2 && e <= 1 && tr <= 1
        # Rule 4: Decreasing + low antecedent + no tail risk
        [0.65, 0.30, 0.05, 0.0, 0.0]
    elseif a <= 1 && e >= 3
        # Rule 5: High forecast but dry
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

    # Forecast agreement modifier
    uniform = fill(0.20, 5)
    if ag == 0       # Low agreement → more uncertainty
        probs = 0.5 .* probs .+ 0.5 .* uniform
    elseif ag == 1   # Medium agreement
        probs = 0.8 .* probs .+ 0.2 .* uniform
    end

    return probs ./ sum(probs)
end

"""
Build the full risk CPT as a 3D tensor for DiscreteTransition.

RxInfer DiscreteTransition expects a transition matrix where:
  T[child_state, parent_combo] maps a flattened parent index to child distribution.

Since risk_level has 5 parents, we flatten them into a single "super-parent"
and build a (5 × N_combos) matrix.

Returns: (risk_tensor, n_combos)
"""
function build_risk_cpt(; include_agreement::Bool=true, include_tail_risk::Bool=false)
    if include_tail_risk
        if include_agreement
            n_combos = 5 * 5 * 3 * 3 * 3 * 4  # 2700
        else
            n_combos = 5 * 5 * 3 * 3 * 4      # 900
        end
    else
        if include_agreement
            n_combos = 5 * 5 * 3 * 3 * 3  # 675
        else
            n_combos = 5 * 5 * 3 * 3      # 225
        end
    end

    cpt = zeros(Float64, 5, n_combos)
    idx = 0

    if include_tail_risk && include_agreement
        for tl in 1:4, ag in 1:3, tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
            idx += 1
            cpt[:, idx] = compute_risk_probs(ant, ex, sp, tr, ag, tl)
        end
    elseif include_tail_risk
        for tl in 1:4, tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
            idx += 1
            cpt[:, idx] = compute_risk_probs(ant, ex, sp, tr, 3, tl)
        end
    elseif include_agreement
        for ag in 1:3, tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
            idx += 1
            cpt[:, idx] = compute_risk_probs(ant, ex, sp, tr, ag, 1)
        end
    else
        for tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
            idx += 1
            cpt[:, idx] = compute_risk_probs(ant, ex, sp, tr, 3, 1)
        end
    end

    return cpt, n_combos
end

"""
Build the action CPT (4 × 5 matrix): action | risk_level.
"""
function build_action_cpt()::Matrix{Float64}
    # Rows = action states, Cols = risk_level states
    # Matches Python exactly
    return [
        0.95  0.15  0.00  0.00  0.00;  # Monitor
        0.05  0.80  0.20  0.05  0.00;  # Alert
        0.00  0.05  0.75  0.25  0.05;  # Prepare
        0.00  0.00  0.05  0.70  0.95;  # Act
    ]
end

# ============================================================================
# RxInfer MODEL DEFINITION
# ============================================================================

#=
RxInfer.jl uses the @model macro to define probabilistic models.
For discrete BNs with fixed CPTs we use:
  - Categorical(prior) for root nodes
  - DiscreteTransition(parent, cpt) for conditional nodes

Since risk_level depends on 5 parents, we encode the joint parent state
as a single Categorical variable via a deterministic mapping, then use
DiscreteTransition with the full CPT.

NOTE: RxInfer's DiscreteTransition(y, x, T) means:
  y ~ Categorical(T * x)  where x is a one-hot vector.
=#

"""
Encode multiple discrete parent indices into a single flat index (1-based).
Used to collapse 5 parents into one "super-parent" for DiscreteTransition.
"""
function encode_parents(ant::Int, exc::Int, spa::Int, tre::Int, agr::Int;
                        tail::Int=1, include_tail_risk::Bool=false)::Int
    if include_tail_risk
        # Order: tail(outer) > agreement > trend > spatial > exceedance > antecedent(inner)
        return ((tail - 1) * 3 * 3 * 3 * 5 * 5 +
                (agr - 1) * 3 * 3 * 5 * 5 +
                (tre - 1) * 3 * 5 * 5 +
                (spa - 1) * 5 * 5 +
                (exc - 1) * 5 +
                (ant - 1)) + 1
    else
        return ((agr - 1) * 3 * 3 * 5 * 5 +
                (tre - 1) * 3 * 5 * 5 +
                (spa - 1) * 5 * 5 +
                (exc - 1) * 5 +
                (ant - 1)) + 1
    end
end

function encode_parents_no_agreement(ant::Int, exc::Int, spa::Int, tre::Int;
                                     tail::Int=1, include_tail_risk::Bool=false)::Int
    if include_tail_risk
        # Order: tail(outer) > trend > spatial > exceedance > antecedent(inner)
        return ((tail - 1) * 3 * 3 * 5 * 5 +
                (tre - 1) * 3 * 5 * 5 +
                (spa - 1) * 5 * 5 +
                (exc - 1) * 5 +
                (ant - 1)) + 1
    else
        return ((tre - 1) * 3 * 5 * 5 +
                (spa - 1) * 5 * 5 +
                (exc - 1) * 5 +
                (ant - 1)) + 1
    end
end

# ============================================================================
# INFERENCE (Direct matrix multiplication - no need for full RxInfer model
# for this fixed-CPT case, but we show both approaches)
# ============================================================================

"""
Simple direct inference using matrix multiplication.
For fixed expert CPTs this is equivalent to variable elimination.

Args:
    antecedent_idx: 1-5
    exceedance_idx: 1-5
    spatial_idx: 1-3
    trend_idx: 1-3
    agreement_idx: 1-3
    risk_cpt: (5 × N) risk CPT matrix
    action_cpt: (4 × 5) action CPT matrix

Returns:
    (risk_probs, action_probs) - probability vectors
"""
function infer_direct(
    antecedent_idx::Int,
    exceedance_idx::Int,
    spatial_idx::Int,
    trend_idx::Int,
    agreement_idx::Int,
    risk_cpt::Matrix{Float64},
    action_cpt::Matrix{Float64};
    include_agreement::Bool=true,
    tail_risk_idx::Int=1,
    include_tail_risk::Bool=false,
)
    parent_idx = if include_agreement
        encode_parents(antecedent_idx, exceedance_idx, spatial_idx, trend_idx, agreement_idx;
                        tail=tail_risk_idx, include_tail_risk)
    else
        encode_parents_no_agreement(antecedent_idx, exceedance_idx, spatial_idx, trend_idx;
                                     tail=tail_risk_idx, include_tail_risk)
    end

    # P(risk | parents) = column of CPT
    risk_probs = risk_cpt[:, parent_idx]

    # P(action | evidence) = action_cpt * risk_probs (marginalize over risk)
    action_probs = action_cpt * risk_probs

    return risk_probs, action_probs
end

# ============================================================================
# RxInfer MODEL (reactive message-passing version)
#
# This defines the model using RxInfer's @model macro for cases where you
# want to do more sophisticated inference (e.g., learning CPTs from data,
# online/streaming updates, or handling missing evidence).
# ============================================================================

if HAS_RXINFER
    @eval @model function flood_bn_model(; risk_cpt_matrix, action_cpt_matrix, n_parent_combos)
        parent_combo ~ Categorical(fill(1.0 / n_parent_combos, n_parent_combos))
        risk_level ~ DiscreteTransition(parent_combo, risk_cpt_matrix)
        action ~ DiscreteTransition(risk_level, action_cpt_matrix)
    end

    @eval function infer_rxinfer(
        antecedent_idx::Int, exceedance_idx::Int, spatial_idx::Int,
        trend_idx::Int, agreement_idx::Int; include_agreement::Bool=true,
    )
        risk_cpt, n_combos = build_risk_cpt(; include_agreement)
        action_cpt = build_action_cpt()

        parent_idx = if include_agreement
            encode_parents(antecedent_idx, exceedance_idx, spatial_idx, trend_idx, agreement_idx)
        else
            encode_parents_no_agreement(antecedent_idx, exceedance_idx, spatial_idx, trend_idx)
        end

        parent_evidence = zeros(n_combos)
        parent_evidence[parent_idx] = 1.0

        result = infer(
            model = flood_bn_model(;
                risk_cpt_matrix  = risk_cpt,
                action_cpt_matrix = action_cpt,
                n_parent_combos  = n_combos,
            ),
            data = (parent_combo = parent_evidence,),
        )
        return probvec(result.posteriors[:risk_level]), probvec(result.posteriors[:action])
    end
end

# ============================================================================
# BOUNDARY PROCESSING
# ============================================================================

"""
Boundary data as a named tuple / struct for type safety.
"""
struct BoundaryInput
    id::String
    name::String
    country::String
    antecedent_rainfall_mm::Float64
    antecedent_category::String
    rainfall_trend::String
    gefs_eprob_heavy::Float64
    spatial_coverage::Float64
    forecast_agreement::String
    ens_max_ratio::Float64
end

struct BoundaryResult
    boundary_id::String
    boundary_name::String
    country::String
    antecedent_category::String
    rainfall_trend::String
    risk_level::String
    risk_probabilities::Vector{Float64}
    recommended_action::String
    action_probabilities::Vector{Float64}
    confidence::Float64
end

"""
Process a single boundary through the BN.
"""
function process_boundary(
    b::BoundaryInput,
    risk_cpt::Matrix{Float64},
    action_cpt::Matrix{Float64};
    include_agreement::Bool=true,
    include_tail_risk::Bool=false,
)::BoundaryResult
    ant_idx = categorize_antecedent(b.antecedent_rainfall_mm)
    exc_idx = categorize_exceedance(b.gefs_eprob_heavy)
    spa_idx = categorize_spatial(b.spatial_coverage)
    tre_idx = categorize_trend(b.rainfall_trend)
    agr_idx = categorize_agreement(b.forecast_agreement)
    tl_idx  = categorize_tail_risk(b.ens_max_ratio)

    risk_probs, action_probs = infer_direct(
        ant_idx, exc_idx, spa_idx, tre_idx, agr_idx,
        risk_cpt, action_cpt;
        include_agreement,
        tail_risk_idx=tl_idx,
        include_tail_risk,
    )

    risk_idx = argmax(risk_probs)
    action_idx = argmax(action_probs)

    return BoundaryResult(
        b.id,
        b.name,
        b.country,
        ANTECEDENT_STATES[ant_idx],
        TREND_STATES[tre_idx],
        RISK_STATES[risk_idx],
        risk_probs,
        ACTION_STATES[action_idx],
        action_probs,
        maximum(action_probs),
    )
end

"""
Process all boundaries. Pre-builds CPTs once for efficiency.
"""
function process_all_boundaries(
    boundaries::Vector{BoundaryInput};
    include_agreement::Bool=true,
    include_tail_risk::Bool=false,
)::Vector{BoundaryResult}
    risk_cpt, _ = build_risk_cpt(; include_agreement, include_tail_risk)
    action_cpt = build_action_cpt()

    results = Vector{BoundaryResult}(undef, length(boundaries))

    for (i, b) in enumerate(boundaries)
        results[i] = process_boundary(b, risk_cpt, action_cpt; include_agreement, include_tail_risk)
        if i % 50 == 0
            @info "Processed $i/$(length(boundaries)) boundaries"
        end
    end

    return results
end

# ============================================================================
# CSV / DataFrame integration (optional, requires CSV + DataFrames)
# ============================================================================

"""
Convert results to a DataFrame (requires DataFrames.jl).
"""
function results_to_dataframe(results::Vector{BoundaryResult})
    try
        @eval using DataFrames
    catch
        @warn "DataFrames.jl not available, returning raw results"
        return results
    end

    return DataFrames.DataFrame(
        boundary_id        = [r.boundary_id for r in results],
        boundary_name      = [r.boundary_name for r in results],
        country            = [r.country for r in results],
        antecedent_category = [r.antecedent_category for r in results],
        rainfall_trend     = [r.rainfall_trend for r in results],
        risk_level         = [r.risk_level for r in results],
        recommended_action = [r.recommended_action for r in results],
        confidence         = [r.confidence for r in results],
    )
end

# ============================================================================
# DEMO / SELF-TEST
# ============================================================================

"""
Run a quick self-test to verify CPTs match the Python implementation.
"""
function self_test()
    @info "Running self-test..."

    risk_cpt, _ = build_risk_cpt(; include_agreement=true)
    action_cpt = build_action_cpt()

    # Test case 1: Saturated + Very_High + Widespread + Increasing + High agreement
    # Should give Extreme risk
    rp, ap = infer_direct(5, 5, 3, 3, 3, risk_cpt, action_cpt)
    @info "Test 1 (worst case):" risk=RISK_STATES[argmax(rp)] action=ACTION_STATES[argmax(ap)]
    @assert RISK_STATES[argmax(rp)] == "Extreme" "Expected Extreme risk"
    @assert ACTION_STATES[argmax(ap)] == "Act" "Expected Act action"

    # Test case 2: Dry + Very_Low + Localized + Decreasing + High agreement
    # Should give Minimal risk
    rp2, ap2 = infer_direct(1, 1, 1, 1, 3, risk_cpt, action_cpt)
    @info "Test 2 (best case):" risk=RISK_STATES[argmax(rp2)] action=ACTION_STATES[argmax(ap2)]
    @assert RISK_STATES[argmax(rp2)] == "Minimal" "Expected Minimal risk"
    @assert ACTION_STATES[argmax(ap2)] == "Monitor" "Expected Monitor action"

    # Test case 3: Normal + Medium + Moderate + Stable + Medium agreement
    rp3, ap3 = infer_direct(2, 3, 2, 2, 2, risk_cpt, action_cpt)
    @info "Test 3 (moderate):" risk=RISK_STATES[argmax(rp3)] action=ACTION_STATES[argmax(ap3)]

    # Test case 4: Low agreement should spread probabilities
    rp_high, _ = infer_direct(3, 3, 2, 2, 3, risk_cpt, action_cpt)
    rp_low, _ = infer_direct(3, 3, 2, 2, 1, risk_cpt, action_cpt)
    entropy_high = -sum(p * log(max(p, 1e-10)) for p in rp_high)
    entropy_low  = -sum(p * log(max(p, 1e-10)) for p in rp_low)
    @info "Test 4 (agreement effect):" entropy_high entropy_low
    @assert entropy_low > entropy_high "Low agreement should increase entropy"

    @info "All self-tests passed!"
end

# ============================================================================
# CLI ENTRY POINT
# ============================================================================

"""
Parse a flag value from ARGS: --flag value → value (or nothing).
"""
function getarg(flag::String)
    i = findfirst(==(flag), ARGS)
    return i === nothing || i == length(ARGS) ? nothing : ARGS[i + 1]
end

"""
Run CSV-driven inference. Reads a prep CSV (one row per boundary) and writes
a result CSV with the full risk + action probability vectors.
"""
function run_csv(input_csv::String, output_csv::String;
                 include_agreement::Bool, include_tail_risk::Bool)
    df = CSV.read(input_csv, DataFrames.DataFrame)

    has_ratio = "ens_max_ratio" in names(df)
    if include_tail_risk && !has_ratio
        @warn "--tail-risk requested but ens_max_ratio column not in CSV; disabling"
        include_tail_risk = false
    end

    inputs = Vector{BoundaryInput}(undef, DataFrames.nrow(df))
    for (i, row) in enumerate(DataFrames.eachrow(df))
        inputs[i] = BoundaryInput(
            String(row.id),
            String(row.name),
            String(row.country),
            Float64(row.antecedent_rainfall_mm),
            "",
            String(row.rainfall_trend),
            Float64(row.gefs_eprob_heavy),
            Float64(row.spatial_coverage),
            String(row.forecast_agreement),
            has_ratio ? Float64(row.ens_max_ratio) : 0.0,
        )
    end

    @info "Processing $(length(inputs)) boundaries (agreement=$include_agreement, tail_risk=$include_tail_risk)"
    results = process_all_boundaries(inputs; include_agreement, include_tail_risk)

    out = DataFrames.DataFrame(
        boundary_id         = [r.boundary_id for r in results],
        boundary_name       = [r.boundary_name for r in results],
        country             = [r.country for r in results],
        antecedent_category = [r.antecedent_category for r in results],
        rainfall_trend      = [r.rainfall_trend for r in results],
        risk_level          = [r.risk_level for r in results],
        recommended_action  = [r.recommended_action for r in results],
        confidence          = [r.confidence for r in results],
        risk_minimal        = [r.risk_probabilities[1] for r in results],
        risk_low            = [r.risk_probabilities[2] for r in results],
        risk_moderate       = [r.risk_probabilities[3] for r in results],
        risk_high           = [r.risk_probabilities[4] for r in results],
        risk_extreme        = [r.risk_probabilities[5] for r in results],
        action_monitor      = [r.action_probabilities[1] for r in results],
        action_alert        = [r.action_probabilities[2] for r in results],
        action_prepare      = [r.action_probabilities[3] for r in results],
        action_act          = [r.action_probabilities[4] for r in results],
    )

    mkpath(dirname(abspath(output_csv)))
    CSV.write(output_csv, out)
    @info "Wrote $output_csv rows=$(DataFrames.nrow(out))"

    # Brief distribution print
    risk_counts = DataFrames.combine(DataFrames.groupby(out, :risk_level), DataFrames.nrow => :n)
    @info "Risk distribution:" risk_counts
    action_counts = DataFrames.combine(DataFrames.groupby(out, :recommended_action), DataFrames.nrow => :n)
    @info "Action distribution:" action_counts
end

function main()
    if "--test" in ARGS
        self_test()
        return
    end

    input_csv = getarg("--input-csv")
    output_csv = getarg("--output-csv")
    include_agreement = !("--no-agreement" in ARGS)
    include_tail_risk = "--tail-risk" in ARGS

    if input_csv !== nothing && output_csv !== nothing
        run_csv(input_csv, output_csv; include_agreement, include_tail_risk)
        return
    end

    @info "Flood BN IBF v1 (Julia/RxInfer port)"
    @info "Usage: julia flood_bn_ibf_v1.jl --input-csv IN.csv --output-csv OUT.csv [--no-agreement] [--tail-risk]"
    @info "       julia flood_bn_ibf_v1.jl --test"

    b = BoundaryInput(
        "KEN.1", "Nairobi", "Kenya",
        45.0, "Wet", "Increasing",
        0.65, 0.4, "High", 1.5,
    )
    risk_cpt, _ = build_risk_cpt()
    action_cpt = build_action_cpt()
    result = process_boundary(b, risk_cpt, action_cpt)
    @info "Demo result:" boundary=result.boundary_id risk=result.risk_level action=result.recommended_action confidence=@sprintf("%.2f", result.confidence)

    println("\nRisk probabilities:")
    for (state, prob) in zip(RISK_STATES, result.risk_probabilities)
        bar = repeat("█", round(Int, prob * 40))
        @printf("  %-10s %5.1f%% %s\n", state, prob * 100, bar)
    end

    println("\nAction probabilities:")
    for (state, prob) in zip(ACTION_STATES, result.action_probabilities)
        bar = repeat("█", round(Int, prob * 40))
        @printf("  %-10s %5.1f%% %s\n", state, prob * 100, bar)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
