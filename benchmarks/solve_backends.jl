using FixedEffects, Random
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
x_simple = rand(N)

# Hard problem: N=800k, worker-firm (400k × 50k)
N = 800_000
M = 400_000
O = 50_000
Random.seed!(1234)
pid = rand(1:M, N)
fid = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in pid]
x_hard = rand(N)
fes_hard = [FixedEffect(pid), FixedEffect(fid)]
y = rand(N)
fes_interact = [FixedEffect(pid), FixedEffect(pid; interaction = y)]

##############################################################################
# CPU
##############################################################################

println("Simple (N=10M, 100k×100), first run:") # ~3 s
@time solve_residuals!(deepcopy(x_simple), fes_simple)
println("Simple (N=10M, 100k×100), second run:") # ~0.5 s
@time solve_residuals!(deepcopy(x_simple), fes_simple)

println("Hard (N=800k, 400k×50k), Float32, first run:") # ~3 s
@time solve_residuals!(deepcopy(x_hard), fes_hard; double_precision = false)
println("Hard (N=800k, 400k×50k), Float32, second run:") # ~2.5 s
@time solve_residuals!(deepcopy(x_hard), fes_hard; double_precision = false)

println("Hard (N=800k, 400k×50k), maxiter=300, first run:") # ~2.5 s
@time solve_residuals!(deepcopy(x_hard), fes_hard; maxiter = 300)
println("Hard (N=800k, 400k×50k), maxiter=300, second run:") # ~2.5 s
@time solve_residuals!(deepcopy(x_hard), fes_hard; maxiter = 300)

println("Hard (N=800k, interacted), first run:") # ~3.5 s
@time solve_residuals!(deepcopy(x_hard), fes_interact; maxiter = 300)
println("Hard (N=800k, interacted), second run:") # ~3 s
@time solve_residuals!(deepcopy(x_hard), fes_interact; maxiter = 300)

##############################################################################
# CUDA
##############################################################################
if isdefined(Main, :CUDA) && CUDA.functional()
    println("Simple (N=10M, 100k×100), CUDA, first run:")
    @time solve_residuals!(deepcopy(x_simple), fes_simple; method = :CUDA)
    println("Simple (N=10M, 100k×100), CUDA, second run:")
    @time solve_residuals!(deepcopy(x_simple), fes_simple; method = :CUDA)

    println("Hard (N=800k, 400k×50k), CUDA, first run:")
    @time solve_residuals!(deepcopy(x_hard), fes_hard; method = :CUDA)
    println("Hard (N=800k, 400k×50k), CUDA, second run:")
    @time solve_residuals!(deepcopy(x_hard), fes_hard; method = :CUDA)
end

##############################################################################
# Metal
##############################################################################
if isdefined(Main, :Metal) && Metal.functional()
    println("Simple (N=10M, 100k×100), Metal, first run:") # ~18 s
    @time solve_residuals!(Float32.(deepcopy(x_simple)), fes_simple; method = :Metal, double_precision = false)
    println("Simple (N=10M, 100k×100), Metal, second run:") # ~1.5 s
    @time solve_residuals!(Float32.(deepcopy(x_simple)), fes_simple; method = :Metal, double_precision = false)

    println("Hard (N=800k, 400k×50k), Metal, first run:") # ~3.3 s
    @time solve_residuals!(Float32.(deepcopy(x_hard)), fes_hard; method = :Metal, double_precision = false, maxiter = 300)
    println("Hard (N=800k, 400k×50k), Metal, second run:") # ~1.6 s
    @time solve_residuals!(Float32.(deepcopy(x_hard)), fes_hard; method = :Metal, double_precision = false, maxiter = 300)
end
