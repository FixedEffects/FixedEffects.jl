using Test

using FixedEffects, StatsModels, DataFrames, Random, FillArrays

Random.seed!(789)
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)


method_s = [:lsmr, :lsmr_threads, :lsmr_parallel]
if Base.USE_GPL_LIBS
	push!(method_s, :cholesky, :qr)
end
try 
    using CuArrays, GPUArrays
    push!(method_s, :lsmr_gpu)
    GPUArrays.allowscalar(false)
catch e
    @info "CuArrays not found, skipping test of :lsmr_gpu"
end


dat = DataFrame(x=x, p1=categorical(p1), p2=categorical(p2))
X = ModelMatrix(ModelFrame(@formula(x ~ p1 + p2), dat)).m
r_ols = x - X*( (X'*X) \ (X'*x) )
 

fes = [FixedEffect(p1), FixedEffect(p2)]

c_lsmr,_,_ = solve_coefficients!(copy(x), deepcopy(fes), method=:lsmr)


for method in method_s
    @testset "$method" begin        
	    (c, iter, conv) = solve_coefficients!(copy(x), deepcopy(fes),
                                              method = method)
        @test c ≈ c_lsmr
	    (r, iter, conv) = solve_residuals!(copy(x),deepcopy(fes),
                                           method = method)
        @test r ≈ r_ols
        R = 5
        (r, iter, conv) = solve_residuals!(repeat(x, outer=[1, R]),
                                           deepcopy(fes), method=method)
        @test r ≈ repeat(r_ols, outer=[1,R])
    end
end
if :lsmr_gpu ∈ method_s
    @testset "lsmr_gpu Float32" begin
        method = :lsmr_gpu
        (c, iter, conv) = solve_coefficients!(copy(Float32.(x)), deepcopy(fes),
                                              method=method) 
        @test c ≈ c_lsmr
	    (r, iter, conv) = solve_residuals!(copy(Float32.(x)),deepcopy(fes),
                                           method=method)
        @test r ≈ Float32.(r_ols)
        R = 5
        (r, iter, conv) = solve_residuals!(repeat(Float32.(x), outer=[1, R]),
                                           deepcopy(fes),
                                           method=method)  
        @test r ≈ repeat(Float32.(r_ols), outer=[1,R])
    end
end
