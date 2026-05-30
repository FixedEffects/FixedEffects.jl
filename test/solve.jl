

p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = [ 0.5548445405298847 , 0.9444014472663531 , 0.0510866660400604 , 0.9415750229576445 , 0.697755708534771  , 0.9664962514198971 , 0.12752269572311858, 0.4633531422366297 , 0.03341608526498096, 0.1647934493047556]
fes = [FixedEffect(p1), FixedEffect(p2)]
r_ols =  [-0.2015993617092453,  0.2015993617092464, -0.2015993617092463,  0.2015993617092462, -0.2015993617092465,  0.2015993617092467, -0.2015993617092465,  0.2015993617092470, -0.2015993617092468,  0.20159936170924628]
# test solve_coefficients!
(coefs, iter, conv) = solve_coefficients!(deepcopy(x), fes)
@test conv
# verify coefficients reproduce OLS residuals: x - sum of FE coefficients ≈ r_ols
@test x .- coefs[1] .- coefs[2] ≈ r_ols

# test solve_residuals!
(r, iter, conv) = solve_residuals!(deepcopy(x), fes)
@test r ≈ r_ols

@testset "maxiter semantics" begin
	(r0, iter0, conv0) = @test_logs (:warn, r"solve_residuals!") solve_residuals!(deepcopy(x), fes; maxiter = 0)
	@test iter0 == 0
	@test !conv0
	(r1, iter1, conv1) = @test_logs (:warn, r"solve_residuals!") solve_residuals!(deepcopy(x), fes; maxiter = 1)
	@test iter1 == 1
	(coefs1, coef_iter1, coef_conv1) = @test_logs (:warn, r"solve_coefficients!") solve_coefficients!(deepcopy(x), fes; maxiter = 1)
	@test coef_iter1 == 1
	@test_throws ArgumentError solve_residuals!(deepcopy(x), fes; maxiter = -1)
	@test_throws ArgumentError solve_coefficients!(deepcopy(x), fes; maxiter = -1)
end

# PooledArrays
(r, iter, conv) = solve_residuals!(deepcopy(x), [FixedEffect(PooledArray(p1)), FixedEffect(PooledArray(p2))])
@test r ≈ r_ols

# CategorialArrays
(r, iter, conv) = solve_residuals!(deepcopy(x), [FixedEffect(categorical(p1)), FixedEffect(categorical(p2))])
@test r ≈ r_ols


method_s = [:cpu]
if CUDA.functional()
	push!(method_s, :CUDA)
end
if Metal.functional()
	push!(method_s, :Metal)
end
for method in method_s
	println("$method Float32")
	local (r, iter, conv) = solve_residuals!(deepcopy(x), fes, method=method, double_precision = false)
	@test Float32.(r) ≈ Float32.(r_ols)
end

function _residual_from_coefs(y, coefs)
	out = copy(y)
	for coef in coefs
		out .-= coef
	end
	return out
end

@testset "GPU parity" begin
	n_gpu = 2048
	p1_gpu = mod1.(1:n_gpu, 32)
	p2_gpu = mod1.((1:n_gpu) .* 7, 41)
	x_gpu = sin.((1:n_gpu) ./ 3) .+ cos.((1:n_gpu) ./ 11)
	weights_gpu = Weights(1 .+ mod.(1:n_gpu, 5) ./ 10)
	interaction_gpu = 0.5 .+ mod.(1:n_gpu, 11) ./ 13
	fes_gpu = [FixedEffect(p1_gpu), FixedEffect(p2_gpu)]
	fes_interact_gpu = [FixedEffect(p1_gpu, interaction = interaction_gpu), FixedEffect(p2_gpu)]
	fes_bin_gpu = [FixedEffect(p1_gpu)]
	atol_gpu = 1e-3
	rtol_gpu = 1e-3

	for method in filter(!=(:cpu), method_s)
		cpu_r = solve_residuals!(deepcopy(x_gpu), fes_gpu; double_precision = false)[1]
		gpu_r = solve_residuals!(deepcopy(x_gpu), fes_gpu; method = method, double_precision = false)[1]
		@test gpu_r ≈ cpu_r atol=atol_gpu rtol=rtol_gpu

		cpu_weighted_r = solve_residuals!(deepcopy(x_gpu), fes_gpu, weights_gpu; double_precision = false)[1]
		gpu_weighted_r = solve_residuals!(deepcopy(x_gpu), fes_gpu, weights_gpu; method = method, double_precision = false)[1]
		@test gpu_weighted_r ≈ cpu_weighted_r atol=atol_gpu rtol=rtol_gpu

		cpu_interact_r = solve_residuals!(deepcopy(x_gpu), fes_interact_gpu, weights_gpu; double_precision = false)[1]
		gpu_interact_r = solve_residuals!(deepcopy(x_gpu), fes_interact_gpu, weights_gpu; method = method, double_precision = false)[1]
		@test gpu_interact_r ≈ cpu_interact_r atol=atol_gpu rtol=rtol_gpu

		cpu_coefs = solve_coefficients!(deepcopy(x_gpu), fes_gpu, weights_gpu; double_precision = false)[1]
		gpu_coefs = solve_coefficients!(deepcopy(x_gpu), fes_gpu, weights_gpu; method = method, double_precision = false)[1]
		@test _residual_from_coefs(x_gpu, gpu_coefs) ≈ _residual_from_coefs(x_gpu, cpu_coefs) atol=atol_gpu rtol=rtol_gpu

		cpu_bin_r = solve_residuals!(deepcopy(x_gpu), fes_bin_gpu, weights_gpu; double_precision = false)[1]
		gpu_bin_r = solve_residuals!(deepcopy(x_gpu), fes_bin_gpu, weights_gpu; method = method, double_precision = false)[1]
		@test gpu_bin_r ≈ cpu_bin_r atol=atol_gpu rtol=rtol_gpu
	end

	if Metal.functional()
		@test_throws ArgumentError solve_residuals!(deepcopy(x), fes; method = :Metal, double_precision = true)
		@test_throws ArgumentError solve_coefficients!(deepcopy(x), fes; method = :Metal, double_precision = true)
	end
end


fe = FixedEffect([1, 2])
@test_throws "FixedEffects must have the same length as y" ỹ = solve_residuals!(ones(100), [fe])


# test update_weights
weights = ones(10)
fes = [FixedEffect(p1)]
feM = FixedEffects.AbstractFixedEffectSolver{Float64}(fes, Weights(weights), Val{:cpu})
weights = Weights([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
FixedEffects.update_weights!(feM, weights) 
solve_residuals!(deepcopy(x), feM)[1] ≈ solve_residuals!(deepcopy(x), fes, weights)[1]

weights = Weights(reverse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
FixedEffects.update_weights!(feM, weights)
solve_residuals!(deepcopy(x), feM)[1] ≈ solve_residuals!(deepcopy(x), fes, weights)[1]

# test interacted fixed effects
interaction = [0.2, 0.8, 0.3, 0.7, 0.5, 0.5, 0.4, 0.6, 0.1, 0.9]
fes_interact = [FixedEffect(p1, interaction = interaction)]
(r_interact, iter, conv) = solve_residuals!(deepcopy(x), fes_interact)
@test conv

# test interacted + non-interacted FE with same refs
fes_both = [FixedEffect(p1), FixedEffect(p1, interaction = interaction)]
(r_both, iter, conv) = solve_residuals!(deepcopy(x), fes_both)
@test conv
