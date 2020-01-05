using FixedEffects, Random, Statistics
Random.seed!(1234)
N = 10000000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
fes = [FixedEffect(id1), FixedEffect(id2)]
x = rand(N)

# simple problem
@time solve_residuals!(deepcopy(x), fes)