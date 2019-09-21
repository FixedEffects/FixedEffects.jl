using Test

using FixedEffects, GLM, DataFrames, Random

Random.seed!(789)
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)


method_s = [:lsmr, :lsmr_threads, :lsmr_parallel]
if Base.USE_GPL_LIBS
	push!(method_s, :cholesky, :qr)
end
try 
    using CuArrays
    push!(method_s, :lsmr_gpu)
catch e
    @info "CuArrays not found, skipping test of :lsmr_gpu"
end


dat = DataFrame(x=x, p1=categorical(p1), p2=categorical(p2))
ols = lm(@formula(x ~ p1 + p2), dat)
r_glm = residuals(ols)
fes = [FixedEffect(p1), FixedEffect(p2)]
c_lsmr,_,_ = solve_coefficients!(copy(x), deepcopy(fes), method=:lsmr)

for method in method_s
    @testset "$method" begin        
	    (c, iter, conv) = solve_coefficients!(copy(x), deepcopy(fes),
                                              method = method)
        @test c ≈ c_lsmr
	    (r, iter, conv) = solve_residuals!(copy(x),deepcopy(fes),
                                           method = method)
        @test r ≈ r_glm
        R = 5
        (r, iter, conv) = solve_residuals!(repeat(x, outer=[1, R]),
                                           deepcopy(fes), method=method)
        @test r ≈ repeat(r_glm, outer=[1,R])
    end
end


