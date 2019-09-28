using FixedEffects, Random, Statistics
Random.seed!(1234)
N = 10000000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
fes = [FixedEffect(id1), FixedEffect(id2)]
x = rand(N)

# simple problem
X = deepcopy(x)
@time solve_residuals!(X, fes; tol = 1e-6)
#  0.807994 seconds (848 allocations: 311.411 MiB, 10.14% gc time)
X = [x x x x x x x x x x]
@time solve_residuals!(X, fes)
#  6.610115 seconds (8.83 k allocations: 311.572 MiB, 1.54% gc time)
X = [x x x x x x x x x x]
@time solve_residuals!(X,fes, method = :lsmr_threads)
#  5.120222 seconds (9.19 k allocations: 1.210 GiB, 8.59% gc time)

# More complicated problem
N = 800000 # number of observations
M = 400000 # number of workers
O = 50000 # number of firms
Random.seed!(1234)
pid = rand(1:M, N)
fid = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in pid]
x = rand(N)
fes = [FixedEffect(pid), FixedEffect(fid)]
@time solve_residuals!([x x x x], fes; double_precision = false)
# 12.690011 seconds (191.26 k allocations: 67.887 MiB, 0.03% gc time)
@time solve_residuals!([x x x x], fes, method = :lsmr_threads, maxiter = 300)
# 10.660428 seconds (239.89 k allocations: 189.271 MiB, 0.80% gc time)

# cluster (381 iterations)
@time solve_residuals!([x x x x], fes; double_precision = false)
#  11.760708 seconds (292.08 k allocations: 51.847 MiB, 0.95% gc time)
@time solve_residuals!([x x x x], fes, method = :lsmr_gpu, double_precision = false)
#  2.342490 seconds (1.15 M allocations: 90.584 MiB, 5.05% gc time)



