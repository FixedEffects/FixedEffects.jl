using Revise, Metal, FixedEffects, Random, Statistics
Random.seed!(1234)
N = 10000000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
fes = [FixedEffect(id1), FixedEffect(id2)]
x = Float32.(rand(N))

# simple problem
@time solve_residuals!(deepcopy(x), fes)
#   0.654833 seconds (1.99 k allocations: 390.841 MiB, 3.71% gc time)
# it is slow due as gather! is super slow
@time solve_residuals!(deepcopy(x), fes; method = :Metal)

@time solve_residuals!([x x x x], fes)

@time solve_residuals!([x x x x], fes; method = :Metal)
#   2.319452 seconds (7.71 k allocations: 620.016 MiB, 0.36% gc time)

# More complicated problem
N = 800000 # number of observations
M = 400000 # number of workers
O = 50000 # number of firms
Random.seed!(1234)
pid = rand(1:M, N)
fid = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in pid]
x = rand(N)
fes = [FixedEffect(pid), FixedEffect(fid)]
@time solve_residuals!(deepcopy(x), fes; double_precision = false)
# 2.239687 seconds (110.81 k allocations: 35.470 MiB)
@time solve_residuals!(deepcopy(x), fes; maxiter = 300)
# 3.681389 seconds (97.09 k allocations: 62.593 MiB)

@time solve_residuals!([x x x x], fes; double_precision = false)
# 5.253745 seconds (285.98 k allocations: 184.213 MiB, 0.55% gc time)

@time solve_residuals!([x x x x], fes; maxiter = 300)
# 9.889438 seconds (225.38 k allocations: 311.964 MiB, 0.33% gc time)

