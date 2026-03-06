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
