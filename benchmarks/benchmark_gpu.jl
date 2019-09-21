using FixedEffects, CategoricalArrays, Random, CuArrays, BenchmarkTools, Statistics

Random.seed!(1234)
N = 10000000
K = 100
id1 = categorical(Int.(rand(1:(N/K), N)))
id2 = categorical(Int.(rand(1:K, N)))
id3 = categorical(Int.(rand(1:(N/K), N)))

x = rand(N)
X = rand(N, 10)

(r_g, it_g, conv_g)=solve_residuals!(copy(x), deepcopy([FixedEffect(id1), FixedEffect(id2)]), method = :lsmr_gpu)
(r_c, it_c, conv_c)=solve_residuals!(copy(x), deepcopy([FixedEffect(id1), FixedEffect(id2)]), method = :lsmr);


@show r_c ≈ r_g
@show it_c, it_g
@show conv_c, conv_g


println("CPU, size(x)=$(size(x)), 2 fixed effects")
@btime (r_c, it_c, conv_c)=solve_residuals!($x, [FixedEffect(id1), FixedEffect(id2)], method = :lsmr);
println("GPU, size(x)=$(size(x)), 2 fixed effects")
@btime (r_g, it_g, conv_g)=solve_residuals!($x, [FixedEffect(id1), FixedEffect(id2)], method = :lsmr_gpu);


# Note that most of the GPU time is spend transferring memory 
using Profile
Profile.clear()
@profile solve_residuals!(x, [FixedEffect(id1), FixedEffect(id2)], method = :lsmr_gpu);
Profile.print(noisefloor=1.0)

# Multiple x
X = [x x x x x x x x x x]

println("CPU, size(X)=$(size(X)), 2 fixed effects")
@btime (r_c, it_c, conv_c)=solve_residuals!($X, [FixedEffect(id1), FixedEffect(id2)]);
println("GPU, size(X)=$(size(X)), 2 fixed effects")
@btime (r_g, it_g, conv_g)=solve_residuals!($X, [FixedEffect(id1), FixedEffect(id2)], method = :lsmr_gpu);

# Still most GPU time is copying memory (not surprising given how code works)
Profile.clear()
@profile solve_residuals!(X, [FixedEffect(id1), FixedEffect(id2)], method = :lsmr_gpu);
Profile.print(noisefloor=1.0)

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
X = [x x x x]
@time solve_residuals!(X, [FixedEffect(pid), FixedEffect(fid)])
X = [x x x x]
@time solve_residuals!(X, [FixedEffect(pid), FixedEffect(fid)], method = :lsmr_gpu)

