using FixedEffects, CategoricalArrays, Random, Statistics
Random.seed!(1234)
N = 10000000
K = 100
id1 = categorical(Int.(rand(1:(N/K), N)))
id2 = categorical(Int.(rand(1:K, N)))
id3 = categorical(Int.(rand(1:(N/K), N)))

x = rand(N)
X = [x x x x x x x x x x]

# simple problem
@time solve_residuals!(x, [FixedEffect(id1), FixedEffect(id2)])
#    0.385959 seconds (323 allocations: 235.105 MiB)
@time solve_residuals!(X, [FixedEffect(id1), FixedEffect(id2)])
# 2.659090 seconds (2.63 k allocations: 235.155 MiB)
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
pid = categorical(pid)
fid = categorical(fid)
@time solve_residuals!([x x x x], [FixedEffect(pid), FixedEffect(fid)])


@time solve_residuals!([x x x x], [FixedEffect(pid), FixedEffect(fid)], method = :lsmr_threads)

