using FixedEffects, Random, Statistics
Random.seed!(1234)

# Simple problem
N = 10_000_000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
fes = [FixedEffect(id1), FixedEffect(id2)]
x = rand(N)

@time solve_residuals!(deepcopy(x), fes)

# More complicated problem (worker-firm)
N = 800_000
M = 400_000
O = 50_000
Random.seed!(1234)
pid = rand(1:M, N)
fid = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in pid]
x = rand(N)
fes = [FixedEffect(pid), FixedEffect(fid)]

@time solve_residuals!(deepcopy(x), fes; double_precision = false)
@time solve_residuals!(deepcopy(x), fes; maxiter = 300)

# Interacted fixed effects
y = rand(N)
fes = [FixedEffect(pid), FixedEffect(pid; interaction = y)]
@time solve_residuals!(deepcopy(x), fes)


##############################################################################
# Benchmark: simple problem (N=10M, K=100)
##############################################################################
println("="^60)
println("Simple problem: N=10M, two FEs (100k and 100 groups)")
println("="^60)

N = 10_000_000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
fes = [FixedEffect(id1), FixedEffect(id2)]
x = rand(N)

# warmup
solve_residuals!(deepcopy(x), fes)
# benchmark
b1 = @benchmark solve_residuals!(xc, $fes) setup=(xc=deepcopy($x)) evals=1 samples=10
show(stdout, MIME("text/plain"), b1); println()
println("  Median: $(round(median(b1).time/1e6, digits=1)) ms")

##############################################################################
# Benchmark: hard problem (N=800k, worker-firm)
##############################################################################
println("\n", "="^60)
println("Hard problem: N=800k, worker (400k) x firm (50k)")
println("="^60)

N2 = 800_000
M = 400_000
O = 50_000
pid = rand(1:M, N2)
fid = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in pid]
x2 = rand(N2)
fes2 = [FixedEffect(pid), FixedEffect(fid)]

solve_residuals!(deepcopy(x2), fes2)
b2 = @benchmark solve_residuals!(xc, $fes2) setup=(xc=deepcopy($x2)) evals=1 samples=5
show(stdout, MIME("text/plain"), b2); println()
println("  Median: $(round(median(b2).time/1e6, digits=0)) ms")
