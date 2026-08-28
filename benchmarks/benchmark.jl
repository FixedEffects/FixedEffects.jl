##############################################################################
# Benchmarks for solve_residuals! across backends (:cpu, :CUDA, :Metal).
# Problems are defined once; every available backend runs the same
# specifications. Each specification residualizes 6 columns through one
# solver, which is how FixedEffectModels calls this package (first run
# includes compilation).
# Usage: julia --project -t auto benchmarks/benchmark.jl
##############################################################################

using FixedEffects, Random, StatsBase
try using CUDA catch end
try using Metal catch end
Random.seed!(1234)

##############################################################################
# Setup
##############################################################################

# Simple problem: N=10M, two FEs (100k × 100 groups)
N = 10_000_000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
fes_simple = [FixedEffect(id1), FixedEffect(id2)]
cols_simple = [rand(N) for _ in 1:6]

# Hard problem: N=800k, worker-firm (40k × 5k), same construction as the
# "difficult" setup in FixedEffectModels' benchmark/benchmark.jl
N = 800_000
M = 40_000
O = 5_000
pid = rand(1:M, N)
fid = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in pid]
fes_hard = [FixedEffect(pid), FixedEffect(fid)]
cols_hard = [rand(N) for _ in 1:6]

# Interacted fixed effects: one regressor interacted with both hard FEs
z = rand(N)
fes_hard_interact = [FixedEffect(pid), FixedEffect(pid; interaction = z), FixedEffect(fid), FixedEffect(fid; interaction = z)]

# Three-way absorption (worker, firm, year)
yid = rand(1:36, N)
fes_hard_3way = [FixedEffect(pid), FixedEffect(fid), FixedEffect(yid)]

# Large cardinality: N=10M, worker-firm (500k × 50k), banded construction with
# the firm window tuned so each column takes ~20 LSMR iterations; the worker
# coefficient tile exceeds _SORT_TILE_BYTES, so this spec runs on the sorted
# observation layout on every backend
N_large = 10_000_000
M_large = 500_000
O_large = 50_000
pid_large = rand(1:M_large, N_large)
fid_large = [rand(max(1, div(x, 10)-5_000):min(O_large, div(x, 10)+5_000)) for x in pid_large]
fes_large = [FixedEffect(pid_large), FixedEffect(fid_large)]
cols_large = [rand(N_large) for _ in 1:6]

##############################################################################
# CPU
##############################################################################

println("\n", "="^60)
println("Backend: cpu (Float64)")
println("="^60)

feM = AbstractFixedEffectSolver{Float64}(fes_simple, uweights(length(cols_simple[1])), Val{:cpu})
println("Simple (N=10M, 100k×100), 6 columns, first run:")
@time solve_residuals!([copy(c) for c in cols_simple], feM; progress_bar = false)   # ~1.1 s
println("Simple (N=10M, 100k×100), 6 columns, second run:")
@time solve_residuals!([copy(c) for c in cols_simple], feM; progress_bar = false)   # ~0.8 s

feM = AbstractFixedEffectSolver{Float64}(fes_hard, uweights(N), Val{:cpu})
println("Hard (N=800k, 40k×5k), 6 columns, first run:")
@time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~13.6 s
println("Hard (N=800k, 40k×5k), 6 columns, second run:")
@time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~13.8 s

feM = AbstractFixedEffectSolver{Float64}(fes_hard_interact, uweights(N), Val{:cpu})
println("Hard (N=800k, interacted), 6 columns, first run:")
@time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~22 s
println("Hard (N=800k, interacted), 6 columns, second run:")
@time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~26 s

feM = AbstractFixedEffectSolver{Float64}(fes_hard_3way, uweights(N), Val{:cpu})
println("Hard 3-way (N=800k, 40k×5k×36), 6 columns, first run:")
@time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~20.6 s
println("Hard 3-way (N=800k, 40k×5k×36), 6 columns, second run:")
@time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~19.9 s

feM = AbstractFixedEffectSolver{Float64}(fes_large, uweights(N_large), Val{:cpu})
println("Large (N=10M, 500k×50k, sorted layout), 6 columns, first run:")
@time solve_residuals!([copy(c) for c in cols_large], feM; progress_bar = false)   # ~4.4 s
println("Large (N=10M, 500k×50k, sorted layout), 6 columns, second run:")
@time solve_residuals!([copy(c) for c in cols_large], feM; progress_bar = false)   # ~4.1 s

##############################################################################
# Metal
##############################################################################

if isdefined(Main, :Metal) && Metal.functional()
    println("\n", "="^60)
    println("Backend: Metal (Float32)")
    println("="^60)

    feM = AbstractFixedEffectSolver{Float32}(fes_simple, uweights(length(cols_simple[1])), Val{:Metal})
    println("Simple (N=10M, 100k×100), 6 columns, first run:")
    @time solve_residuals!([copy(c) for c in cols_simple], feM; progress_bar = false)   # ~9.9 s
    println("Simple (N=10M, 100k×100), 6 columns, second run:")
    @time solve_residuals!([copy(c) for c in cols_simple], feM; progress_bar = false)   # ~0.38 s

    feM = AbstractFixedEffectSolver{Float32}(fes_hard, uweights(N), Val{:Metal})
    println("Hard (N=800k, 40k×5k), 6 columns, first run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~3.5 s
    println("Hard (N=800k, 40k×5k), 6 columns, second run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~0.75 s

    feM = AbstractFixedEffectSolver{Float32}(fes_hard_interact, uweights(N), Val{:Metal})
    println("Hard (N=800k, interacted), 6 columns, first run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~3.2 s
    println("Hard (N=800k, interacted), 6 columns, second run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~1.3 s

    feM = AbstractFixedEffectSolver{Float32}(fes_hard_3way, uweights(N), Val{:Metal})
    println("Hard 3-way (N=800k, 40k×5k×36), 6 columns, first run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~1.6 s
    println("Hard 3-way (N=800k, 40k×5k×36), 6 columns, second run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)   # ~0.95 s

    feM = AbstractFixedEffectSolver{Float32}(fes_large, uweights(N_large), Val{:Metal})
    println("Large (N=10M, 500k×50k, sorted layout), 6 columns, first run:")
    @time solve_residuals!([copy(c) for c in cols_large], feM; progress_bar = false)   # ~2.8 s
    println("Large (N=10M, 500k×50k, sorted layout), 6 columns, second run:")
    @time solve_residuals!([copy(c) for c in cols_large], feM; progress_bar = false)   # ~1.0 s
end

##############################################################################
# CUDA
##############################################################################

if isdefined(Main, :CUDA) && CUDA.functional()
    println("\n", "="^60)
    println("Backend: CUDA (Float32)")
    println("="^60)

    feM = AbstractFixedEffectSolver{Float32}(fes_simple, uweights(length(cols_simple[1])), Val{:CUDA})
    println("Simple (N=10M, 100k×100), 6 columns, first run:")
    @time solve_residuals!([copy(c) for c in cols_simple], feM; progress_bar = false)
    println("Simple (N=10M, 100k×100), 6 columns, second run:")
    @time solve_residuals!([copy(c) for c in cols_simple], feM; progress_bar = false)

    feM = AbstractFixedEffectSolver{Float32}(fes_hard, uweights(N), Val{:CUDA})
    println("Hard (N=800k, 40k×5k), 6 columns, first run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)
    println("Hard (N=800k, 40k×5k), 6 columns, second run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)

    feM = AbstractFixedEffectSolver{Float32}(fes_hard_interact, uweights(N), Val{:CUDA})
    println("Hard (N=800k, interacted), 6 columns, first run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)
    println("Hard (N=800k, interacted), 6 columns, second run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)

    feM = AbstractFixedEffectSolver{Float32}(fes_hard_3way, uweights(N), Val{:CUDA})
    println("Hard 3-way (N=800k, 40k×5k×36), 6 columns, first run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)
    println("Hard 3-way (N=800k, 40k×5k×36), 6 columns, second run:")
    @time solve_residuals!([copy(c) for c in cols_hard], feM; progress_bar = false)

    feM = AbstractFixedEffectSolver{Float32}(fes_large, uweights(N_large), Val{:CUDA})
    println("Large (N=10M, 500k×50k, sorted layout), 6 columns, first run:")
    @time solve_residuals!([copy(c) for c in cols_large], feM; progress_bar = false)
    println("Large (N=10M, 500k×50k, sorted layout), 6 columns, second run:")
    @time solve_residuals!([copy(c) for c in cols_large], feM; progress_bar = false)
end
