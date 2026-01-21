using Revise, Metal, FixedEffects, Random, Statistics
Random.seed!(1234)
N = 10000000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
fes = [FixedEffect(id1), FixedEffect(id2)]
x = Float32.(rand(N))


# here what takes time if the seocnd fixede ffects where K is very small and so there is a lot of trheads that want to write on the same thing. In that case, it would probably be good to actually pre-compute permutation for each fixed effects once, and then do as manu groups as permutations etc


# simple problem
@time solve_residuals!(deepcopy(x), fes; double_precision = false)
#   0.654833 seconds (1.99 k allocations: 390.841 MiB, 3.71% gc time)
@time solve_residuals!(deepcopy(x), fes; double_precision = false, method = :Metal)
#   0.298326 seconds (129.08 k allocations: 79.208 MiB)
@time solve_residuals!([x x x x], fes)
#   1.604061 seconds (1.25 M allocations: 416.364 MiB, 4.21% gc time, 30.57% compilation time)
@time solve_residuals!([x x x x], fes; method = :Metal)
#   0.790909 seconds (531.78 k allocations: 204.363 MiB, 3.19% compilation time)



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
#   8.294446 seconds (225.13 k allocations: 67.777 MiB, 0.11% gc time)

@time solve_residuals!([x x x x], fes; double_precision = false, method = :Metal)
#   1.605953 seconds (3.25 M allocations: 103.342 MiB, 1.82% gc time)


