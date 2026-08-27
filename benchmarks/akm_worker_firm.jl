using FixedEffects
using LinearAlgebra
using Printf
using Random
using StatsBase

# Simulated counterpart to the xhdfe AKM-style performance example:
# log wage on seniority controls, absorbing worker, firm, and year fixed effects.
# This package is the residualization backend, so worker-clustered SEs are out of scope here.
# Usage: julia --project -t auto benchmarks/akm_worker_firm.jl [--seed=1234] [--method=cpu]
#        [--double-precision=true] [--tol=1e-8] [--maxiter=Inf]
const N_WORKERS = 50_000
const N_FIRMS = 7_000
const N_YEARS = 36
const OBS_PER_WORKER = 8
const FIRM_WINDOW = 10
const MOVE_PROBABILITY = 0.35
const CONTROL_NAMES = ("tenure", "tenure_sq", "experience", "experience_sq", "mover")

struct AKMPanel
    worker::Vector{Int32}
    firm::Vector{Int32}
    year::Vector{Int32}
    y::Vector{Float64}
    x::Matrix{Float64}
end

Base.length(panel::AKMPanel) = length(panel.y)

function parse_options(args)
    options = Dict{String,String}()

    for arg in args
        startswith(arg, "--") || error("Unexpected positional argument '$arg'. Use --key=value options.")
        key_value = split(arg[3:end], "=", limit = 2)
        length(key_value) == 2 || error("Expected --key=value, got $arg")
        options[key_value[1]] = key_value[2]
    end

    return options
end

option(options, key, default::Int) = parse(Int, get(options, key, string(default)))
option(options, key, default::Float64) = parse(Float64, get(options, key, string(default)))

function option(options, key, default::Bool)
    value = lowercase(get(options, key, string(default)))
    value in ("true", "yes", "1") && return true
    value in ("false", "no", "0") && return false
    error("Expected boolean for --$key, got '$value'")
end

function method_option(options)
    value = lowercase(get(options, "method", "cpu"))
    value == "cpu" && return :cpu
    value == "cuda" && return :CUDA
    value == "metal" && return :Metal
    error("Expected --method=cpu, --method=CUDA, or --method=Metal, got '$value'")
end

function maxiter_option(options)
    value = lowercase(get(options, "maxiter", "inf"))
    value in ("inf", "infinity") && return typemax(Int)
    return parse(Int, value)
end

function load_backend!(method::Symbol)
    if method == :CUDA
        @eval using CUDA
        CUDA.functional() || error("CUDA was requested but CUDA.functional() is false")
    elseif method == :Metal
        @eval using Metal
    end
    return nothing
end

function local_firm(rng::AbstractRNG, anchor::Int)
    lo = max(1, anchor - FIRM_WINDOW)
    hi = min(N_FIRMS, anchor + FIRM_WINDOW)
    return rand(rng, lo:hi)
end

function simulate_akm_panel(; seed::Int)
    rng = MersenneTwister(seed)
    n = N_WORKERS * OBS_PER_WORKER
    worker = Vector{Int32}(undef, n)
    firm = Vector{Int32}(undef, n)
    year = Vector{Int32}(undef, n)
    y = Vector{Float64}(undef, n)
    x = Matrix{Float64}(undef, n, length(CONTROL_NAMES))

    worker_effect = randn(rng, N_WORKERS)
    firm_effect = 0.6 .* randn(rng, N_FIRMS)
    year_effect = [0.02 * (t - 1) + 0.05 * sin(2 * pi * (t - 1) / N_YEARS) for t in 1:N_YEARS]

    row = 1
    max_start_year = max(1, N_YEARS - min(OBS_PER_WORKER, N_YEARS) + 1)
    for w in 1:N_WORKERS
        anchor = clamp(1 + fld((w - 1) * N_FIRMS, N_WORKERS), 1, N_FIRMS)
        current_firm = local_firm(rng, anchor)
        start_year = rand(rng, 1:max_start_year)
        base_experience = rand(rng, 1:20)
        tenure = 0

        for spell_t in 1:OBS_PER_WORKER
            moved = spell_t > 1 && rand(rng) < MOVE_PROBABILITY
            if moved
                current_firm = local_firm(rng, anchor)
                tenure = 0
            elseif spell_t > 1
                tenure += 1
            end

            calendar_year = 1 + mod(start_year + spell_t - 2, N_YEARS)
            experience = base_experience + spell_t - 1
            tenure_sq = tenure^2 / 100
            experience_sq = experience^2 / 100
            mover = moved ? 1.0 : 0.0

            worker[row] = Int32(w)
            firm[row] = Int32(current_firm)
            year[row] = Int32(calendar_year)
            x[row, 1] = tenure
            x[row, 2] = tenure_sq
            x[row, 3] = experience
            x[row, 4] = experience_sq
            x[row, 5] = mover
            y[row] = 0.04 * tenure - 0.03 * tenure_sq +
                0.015 * experience - 0.02 * experience_sq +
                0.05 * mover +
                worker_effect[w] + firm_effect[current_firm] +
                year_effect[calendar_year] + 0.2 * randn(rng)
            row += 1
        end
    end

    order = randperm(rng, n)
    return AKMPanel(worker[order], firm[order], year[order], y[order], x[order, :])
end

function akm_estimator_call(panel::AKMPanel;
        method::Symbol,
        double_precision::Bool,
        tol::Real,
        maxiter::Integer)
    y = copy(panel.y)
    x = copy(panel.x)
    fes = [FixedEffect(panel.worker), FixedEffect(panel.firm), FixedEffect(panel.year)]
    T = double_precision ? Float64 : Float32
    solver = AbstractFixedEffectSolver{T}(fes, uweights(T, length(y)), Val{method})
    variables = Vector{AbstractVector{Float64}}(undef, 1 + size(x, 2))
    variables[1] = y
    for j in axes(x, 2)
        variables[j + 1] = view(x, :, j)
    end
    _, iterations, converged = solve_residuals!(variables, solver;
        tol = tol,
        maxiter = maxiter,
        progress_bar = false)
    beta = x \ y
    return (beta = beta, iterations = iterations, converged = converged)
end

function print_result(result)
    @printf("  beta:")
    for (name, value) in zip(CONTROL_NAMES, result.beta)
        @printf(" %s=% .4f", name, value)
    end
    println()
    println("  iterations: ", join(result.iterations, ", "))
    println("  converged:  ", join(result.converged, ", "))
end

options = parse_options(ARGS)
seed = option(options, "seed", 1234)
method = method_option(options)
double_precision = option(options, "double-precision", method == :cpu)
tol = option(options, "tol", double_precision ? 1e-8 : 1e-6)
maxiter = maxiter_option(options)

load_backend!(method)

n = N_WORKERS * OBS_PER_WORKER
println("Simulated AKM benchmark")
println("  observations:    ", n)
println("  worker FE:       ", N_WORKERS)
println("  firm FE:         ", N_FIRMS)
println("  year FE:         ", N_YEARS)
println("  controls:        ", join(CONTROL_NAMES, ", "))
println("  method:          ", method)
println("  double precision:", double_precision)
println("  tol:             ", tol)
println("  maxiter:         ", maxiter == typemax(Int) ? "Inf" : maxiter)
println("  note: this times FE construction, residualization of y and controls, and dense OLS; clustered SEs are not included.")

panel = simulate_akm_panel(; seed = seed)
@printf("  panel memory:    %.1f MiB\n", Base.summarysize(panel) / 2.0^20)

println("\nWarmup")
warmup = akm_estimator_call(panel; method, double_precision, tol, maxiter)
print_result(warmup)

println("\nTimed run")
GC.gc()
timed = @timed akm_estimator_call(panel; method, double_precision, tol, maxiter)
@printf("  time:       %.3f s\n", timed.time)
@printf("  allocated:  %.1f MiB\n", timed.bytes / 2.0^20)
@printf("  gc time:    %.3f s\n", timed.gctime)
print_result(timed.value)
