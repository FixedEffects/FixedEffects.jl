using Metal, FixedEffects, Random, Statistics
Random.seed!(1234)

# Simple problem
N = 10_000_000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
fes = [FixedEffect(id1), FixedEffect(id2)]
x = Float32.(rand(N))

@time solve_residuals!(deepcopy(x), fes; double_precision = false)
@time solve_residuals!(deepcopy(x), fes; double_precision = false, method = :Metal)

# More complicated problem (worker-firm)
N = 8_000_000
M = 4_000_000
O = 500_000
Random.seed!(1234)
pid = rand(1:M, N)
fid = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in pid]
x = rand(N)
fes = [FixedEffect(pid), FixedEffect(fid)]

@time solve_residuals!(deepcopy(x), fes; double_precision = false, maxiter = 100)
@time solve_residuals!(deepcopy(x), fes; double_precision = false, method = :Metal, maxiter = 100)
