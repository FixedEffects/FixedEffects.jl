using Test, FixedEffects

method_s = [:lsmr, :lsmr_threads, :lsmr_parallel]
if Base.USE_GPL_LIBS
	push!(method_s, :cholesky, :qr)
end
try 
    using CuArrays
    push!(method_s, :lsmr_gpu)
    CuArrays.allowscalar(false)
catch e
    @info "CuArrays not found, skipping test of :lsmr_gpu"
end


p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = [ 0.5548445405298847 , 0.9444014472663531 , 0.0510866660400604 , 0.9415750229576445 , 0.697755708534771  , 0.9664962514198971 , 0.12752269572311858, 0.4633531422366297 , 0.03341608526498096, 0.1647934493047556]
fes = [FixedEffect(p1), FixedEffect(p2)]
r_ols =  [-0.2015993617092453,  0.2015993617092464, -0.2015993617092463,  0.2015993617092462, -0.2015993617092465,  0.2015993617092467, -0.2015993617092465,  0.2015993617092470, -0.2015993617092468,  0.20159936170924628]


c_lsmr,_,_ = solve_coefficients!(copy(x), deepcopy(fes), method=:lsmr)


for method in method_s
    @testset "$method" begin        
	    (c, iter, conv) = solve_coefficients!(copy(x), deepcopy(fes),
                                              method = method)
        @test c ≈ c_lsmr
	    (r, iter, conv) = solve_residuals!(copy(x),deepcopy(fes),
                                           method = method)
        @test r ≈ r_ols
        (r, iter, conv) = solve_residuals!([x x x x x],
                                           deepcopy(fes), method=method)
        @test r ≈ [r_ols r_ols r_ols r_ols r_ols]
    end
end
for method in method_s
  if method != :qr
    @testset "$method Float32" begin
        (c, iter, conv) = solve_coefficients!(copy(x), deepcopy(fes),
                                              method=method, double_precision = false) 
        @test c ≈ c_lsmr rtol = 1e-3
	    (r, iter, conv) = solve_residuals!(copy(x),deepcopy(fes),
                                           method=method, double_precision = false)
        @test Float32.(r) ≈ Float32.(r_ols)
        (r, iter, conv) = solve_residuals!([x x x x x],
                                           deepcopy(fes),
                                           method=method, double_precision = false)  
        @test r ≈ [r_ols r_ols r_ols r_ols r_ols] rtol = 1e-3
    end
  end
end
