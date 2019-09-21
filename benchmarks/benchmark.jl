using FixedEffects, CategoricalArrays, Random, Statistics, DataFrames
Random.seed!(1234)
N = 10000000
K = 100
id1 = categorical(Int.(rand(1:(N/K), N)))
id2 = categorical(Int.(rand(1:K, N)))
id3 = categorical(Int.(rand(1:(N/K), N)))

x = rand(N)
X = rand(N, 10)

# simple problem
@time solve_residuals!(x, [FixedEffect(id1), FixedEffect(id2)])
#  0.425070 seconds (138 allocations: 235.102 MiB, 2.92% gc time)
X = [x x x x x x x x x x]
@time solve_residuals!(X, [FixedEffect(id1), FixedEffect(id2)])
# 2.873322 seconds (793 allocations: 235.122 MiB, 0.53% gc time)
X = [x x x x x x x x x x]
@time solve_residuals!(X, [FixedEffect(id1), FixedEffect(id2)], method = :lsmr_threads)
# 5.110589 seconds (702.36 k allocations: 1.071 GiB, 7.72% gc time)

# More complicated problem
N = 8000000 # number of observations
M = 4000000 # number of workers
O = 500000 # number of firms
pid = rand(1:M, N)
fid = zeros(Int64, N)
for i in 1:N
    # low pid matches with low fid
    fid[i] = rand(max(1, div(pid[i], 8)-10):min(O, div(pid[i], 8)+10))
end
x = (pid .* fid .- mean(pid .* fid)) / std(pid .* fid)
CSV.write("/Users/matthieu/ok.csv", DataFrame(pid = pid, fid = fid, x = x))
pid = categorical(pid)
fid = categorical(fid)
X = [x x x x]
@time solve_residuals!(X, [FixedEffect(pid), FixedEffect(fid)])
X = [x x x x]
@time solve_residuals!(X, [FixedEffect(pid), FixedEffect(fid)], method = :lsmr_threads)

