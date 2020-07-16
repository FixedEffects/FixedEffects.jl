using Test, StatsBase, FixedEffects


p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = [ 0.5548445405298847 , 0.9444014472663531 , 0.0510866660400604 , 0.9415750229576445 , 0.697755708534771  , 0.9664962514198971 , 0.12752269572311858, 0.4633531422366297 , 0.03341608526498096, 0.1647934493047556]
fes = [FixedEffect(p1), FixedEffect(p2)]
r_ols =  [-0.2015993617092453,  0.2015993617092464, -0.2015993617092463,  0.2015993617092462, -0.2015993617092465,  0.2015993617092467, -0.2015993617092465,  0.2015993617092470, -0.2015993617092468,  0.20159936170924628]

c_lsmr,_,_ = solve_coefficients!(deepcopy(x), fes)


(c, iter, conv) = solve_coefficients!(deepcopy(x), fes)
@test c ≈ c_lsmr
(r, iter, conv) = solve_residuals!(deepcopy(x), fes)
@test r ≈ r_ols

(c, iter, conv) = solve_residuals!([x x], fes)


# test update_weights
weights = ones(10)
feM = FixedEffects.AbstractFixedEffectSolver{Float64}(fes, Weights(weights), Val{:cpu})
weights = Weights([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
FixedEffects.update_weights!(feM, weights) 
solve_residuals!(deepcopy(x), feM)[1] ≈ solve_residuals!(deepcopy(x), fes, weights)[1]



method_s = [:cpu]
if FixedEffects.has_cuarrays()
	push!(method_s, :gpu)
end
for method in method_s
	println("$method Float32")
	local (c, iter, conv) = solve_coefficients!(deepcopy(x), fes, method=method, double_precision = false) 
	@test c ≈ c_lsmr rtol = 1e-3
	local (r, iter, conv) = solve_residuals!(deepcopy(x),fes, method=method, double_precision = false)
	@test Float32.(r) ≈ Float32.(r_ols)
end


fe = FixedEffect([1, 2])
@test_throws "FixedEffects must have the same length as y" ỹ = solve_residuals!(ones(100), [fe])

