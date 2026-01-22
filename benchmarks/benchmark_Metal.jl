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
#   1.335206 seconds (3.28 M allocations: 402.660 MiB, 1.80% gc time, 123.64% compilation time: <1% of which was recompilation)
@time solve_residuals!([x x x x], fes)
# 1.886616 seconds (3.60 M allocations: 731.777 MiB, 1.34% gc time, 122.12% compilation time: 5% of which was recompilation)
@time solve_residuals!([x x x x], fes; method = :Metal)
#   1.421205 seconds (2.78 M allocations: 497.846 MiB, 1.64% gc time, 110.87% compilation time: <1% of which was recompilation)



# More complicated problem
N = 8000000 # number of observations
M = 4000000 # number of workers
O = 500000 # number of firms
Random.seed!(1234)
pid = rand(1:M, N)
fid = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in pid]
x = rand(N)
fes = [FixedEffect(pid), FixedEffect(fid)]


@time solve_residuals!([x x x x], fes; double_precision = false, maxiter = 100)
# 36.554763 seconds (98.71 M allocations: 5.253 GiB, 1.11% gc time, 114.45% compilation time: 7% of which was recompilation)
@time solve_residuals!([x x x x], fes; double_precision = false, method = :Metal, maxiter = 100)
# 20.652590 seconds (79.33 M allocations: 4.114 GiB, 0.75% gc time, 162.10% compilation time: <1% of which was recompilation)


