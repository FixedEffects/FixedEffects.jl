using CUDA, FixedEffects, Random, BenchmarkTools, Statistics

Random.seed!(1234)

# Simple problem
N = 10_000_000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
x = rand(N)

(r_g, it_g, conv_g) = solve_residuals!(copy(x), [FixedEffect(id1), FixedEffect(id2)], method = :CUDA)
(r_c, it_c, conv_c) = solve_residuals!(copy(x), [FixedEffect(id1), FixedEffect(id2)])

@show r_c ≈ r_g
@show it_c, it_g
@show conv_c, conv_g

println("CPU Float64, N=$N, 2 fixed effects")
@btime solve_residuals!(xc, [FixedEffect($id1), FixedEffect($id2)]) setup=(xc=copy($x))
println("CPU Float32, N=$N, 2 fixed effects")
@btime solve_residuals!(xc, [FixedEffect($id1), FixedEffect($id2)]) setup=(xc=Float32.(copy($x)))
println("GPU Float64, N=$N, 2 fixed effects")
@btime solve_residuals!(xc, [FixedEffect($id1), FixedEffect($id2)], method = :CUDA) setup=(xc=copy($x))
println("GPU Float32, N=$N, 2 fixed effects")
@btime solve_residuals!(xc, [FixedEffect($id1), FixedEffect($id2)], method = :CUDA) setup=(xc=Float32.(copy($x)))

# More complicated problem (worker-firm)
N = 8_000_000
M = 4_000_000
O = 500_000
pid = rand(1:M, N)
fid = [rand(max(1, div(p, 8)-10):min(O, div(p, 8)+10)) for p in pid]
x = rand(N)

println("\nCPU Float64, worker-firm")
@time (rc6, i, c) = solve_residuals!(copy(x), [FixedEffect(pid), FixedEffect(fid)],
                                      tol=sqrt(eps(Float32)))
@show i

println("GPU Float64, worker-firm")
@time (rg6, i, c) = solve_residuals!(copy(x), [FixedEffect(pid), FixedEffect(fid)],
                                      method = :CUDA, tol=sqrt(eps(Float32)))
@show i

println("GPU Float32, worker-firm")
@time (rg3, i, c) = solve_residuals!(Float32.(x), [FixedEffect(pid), FixedEffect(fid)],
                                      method = :CUDA, tol=sqrt(eps(Float32)))
@show i
@show std(rg6 - rg3)
