using LinearAlgebra

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

function _residual_from_coefs(y, fes, coefs)
	out = copy(y)
	for (fe, coef) in zip(fes, coefs)
		if fe.interaction isa UnitWeights
			out .-= coef
		else
			out .-= fe.interaction .* coef
		end
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
	fes_block_gpu = [
		FixedEffect(p1_gpu),
		FixedEffect(p1_gpu, interaction = interaction_gpu),
		FixedEffect(p2_gpu),
		FixedEffect(p2_gpu, interaction = 2 .* interaction_gpu),
	]
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

		cpu_block_r = solve_residuals!(deepcopy(x_gpu), fes_block_gpu, weights_gpu; double_precision = false)[1]
		gpu_block_r = solve_residuals!(deepcopy(x_gpu), fes_block_gpu, weights_gpu; method = method, double_precision = false)[1]
		@test gpu_block_r ≈ cpu_block_r atol=atol_gpu rtol=rtol_gpu

		cpu_coefs = solve_coefficients!(deepcopy(x_gpu), fes_gpu, weights_gpu; double_precision = false)[1]
		gpu_coefs = solve_coefficients!(deepcopy(x_gpu), fes_gpu, weights_gpu; method = method, double_precision = false)[1]
		@test _residual_from_coefs(x_gpu, gpu_coefs) ≈ _residual_from_coefs(x_gpu, cpu_coefs) atol=atol_gpu rtol=rtol_gpu

		cpu_block_coefs = solve_coefficients!(deepcopy(x_gpu), fes_block_gpu, weights_gpu; double_precision = false)[1]
		gpu_block_coefs = solve_coefficients!(deepcopy(x_gpu), fes_block_gpu, weights_gpu; method = method, double_precision = false)[1]
		@test _residual_from_coefs(x_gpu, fes_block_gpu, gpu_block_coefs) ≈
			_residual_from_coefs(x_gpu, fes_block_gpu, cpu_block_coefs) atol=atol_gpu rtol=rtol_gpu

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
feM_compat = FixedEffects.AbstractFixedEffectSolver{Float64}(fes, weights, Val{:cpu}, Base.Threads.nthreads())
@test solve_residuals!(deepcopy(x), feM_compat)[1] ≈ solve_residuals!(deepcopy(x), fes, weights)[1]

# a matrix must be passed as columns (e.g. eachcol), not raw
@test_throws ArgumentError solve_residuals!(rand(10, 2), feM)

# test interacted fixed effects
interaction = [0.2, 0.8, 0.3, 0.7, 0.5, 0.5, 0.4, 0.6, 0.1, 0.9]
fes_interact = [FixedEffect(p1, interaction = interaction)]
(r_interact, iter, conv) = solve_residuals!(deepcopy(x), fes_interact)
@test conv

# test interacted + non-interacted FE with same refs
fes_both = [FixedEffect(p1), FixedEffect(p1, interaction = interaction)]
(r_both, iter, conv) = solve_residuals!(deepcopy(x), fes_both)
@test conv

@testset "block absorption algebra" begin
	id = repeat(1:4, inner = 3)
	slope = [-1.0, 0.0, 1.0, 0.5, 1.0, 1.5, -2.0, -1.0, 0.0, 2.0, 2.0, 2.0]
	y = [2.0, 1.5, 1.0, -1.0, 0.5, 1.5, 3.0, 2.0, 1.0, -2.0, -1.5, -1.0]
	weights = Weights(1 .+ (1:length(y)) ./ 20)
	fes_block = [
		FixedEffect(id),
		FixedEffect(id, interaction = slope),
		FixedEffect(id, interaction = slope .^ 2),
	]
	plan = FixedEffects.AbsorptionPlan(Float64, fes_block, weights)
	@test length(plan.blocks) == 1
	@test FixedEffects.block_width(plan.blocks[1]) == 3
	@test maximum(plan.ranks[1]) == 3
	@test minimum(plan.ranks[1]) < 3

	X = zeros(length(y), 12)
	for i in eachindex(y)
		X[i, id[i]] = 1
		X[i, 4 + id[i]] = slope[i]
		X[i, 8 + id[i]] = slope[i]^2
	end
	sqrtw = sqrt.(weights)
	Xw = X .* reshape(sqrtw, :, 1)
	yw = y .* sqrtw
	r_dense = (yw - Xw * (pinv(Xw) * yw)) ./ sqrtw
	r_block, _, conv = solve_residuals!(copy(y), fes_block, weights)
	@test conv
	@test r_block ≈ r_dense atol = 1e-10

	coefs = solve_coefficients!(copy(y), fes_block, weights)[1]
	@test y .- coefs[1] .- slope .* coefs[2] .- slope.^2 .* coefs[3] ≈ r_block atol = 1e-10

	# several variables through the collection fallback, sharing one solver
	feM_block = FixedEffects.AbstractFixedEffectSolver{Float64}(fes_block, weights, Val{:cpu})
	cols = [copy(y), 2 .* y .+ 1]
	solve_residuals!(cols, feM_block; progress_bar = false)
	@test cols[1] ≈ r_block atol = 1e-10
	@test cols[2] ≈ solve_residuals!(2 .* y .+ 1, fes_block, weights)[1] atol = 1e-10

	id_big = repeat(1:2, inner = 3)
	slope_big = [100_000.0, 100_001.0, 100_002.0, 200_000.0, 200_001.0, 200_002.0]
	y_big = [1.0, 2.0, 4.0, -1.0, 0.5, 3.0]
	fes_big = [FixedEffect(id_big), FixedEffect(id_big, interaction = slope_big)]
	plan_big = FixedEffects.AbsorptionPlan(Float64, fes_big, uweights(length(y_big)))
	@test plan_big.ranks[1] == [2, 2]
	X_big = zeros(length(y_big), 4)
	for i in eachindex(y_big)
		X_big[i, id_big[i]] = 1
		X_big[i, 2 + id_big[i]] = slope_big[i]
	end
	@test solve_residuals!(copy(y_big), fes_big)[1] ≈ y_big - X_big * (pinv(X_big) * y_big) atol = 1e-8

	# The stable moment path also handles weights, a large slope offset, and the
	# less common input order in which the slope precedes the intercept.
	id_stable = repeat(1:2, inner = 3)
	slope_stable = 1.0e7 .+ [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
	weights_stable = Weights([1.0, 2.0, 4.0, 1.5, 2.5, 3.5])
	y_stable = [1.0, -2.0, 4.0, 0.5, 3.0, -1.0]
	X_stable = zeros(length(id_stable), 4)
	for i in eachindex(id_stable)
		g = id_stable[i]
		X_stable[i, g] = 1
		X_stable[i, 2 + g] = slope_stable[i] - slope_stable[firstindex(slope_stable)]
	end
	sqrtw_stable = sqrt.(weights_stable)
	r_stable = (sqrtw_stable .* y_stable -
		(X_stable .* sqrtw_stable) * (pinv(X_stable .* sqrtw_stable) *
		(sqrtw_stable .* y_stable))) ./ sqrtw_stable
	for fes_stable in ([FixedEffect(id_stable), FixedEffect(id_stable, interaction = slope_stable)],
			[FixedEffect(id_stable, interaction = slope_stable), FixedEffect(id_stable)])
		plan_stable = FixedEffects.AbsorptionPlan(Float64, fes_stable, weights_stable)
		@test plan_stable.ranks[1] == [2, 2]
		for g in 1:2
			rows = findall(==(g), id_stable)
			Q = permutedims(plan_stable.qrows[1][:, rows])
			@test Q' * Q ≈ I atol = 1e-12
		end
		@test solve_residuals!(copy(y_stable), fes_stable, weights_stable)[1] ≈
			r_stable atol = 1e-10
	end

	# Two slopes without an intercept still use the general block path.
	slope2 = [1.0, 2.0, 4.0, 2.0, 3.0, 5.0]
	fes_slopes = [FixedEffect(id_stable, interaction = slope2),
		FixedEffect(id_stable, interaction = slope2 .^ 2)]
	plan_slopes = FixedEffects.AbsorptionPlan(Float64, fes_slopes, weights_stable)
	@test plan_slopes.ranks[1] == [2, 2]
end

# Independent implementation of the exact one-block projection residual,
# y ← y - Z G⁺ Z' W y with G⁺ = R R' from the plan transforms: the oracle the
# operator tests below are checked against.
function project_block!(y::AbstractVector, block, transform, weights)
	k = FixedEffects.block_width(block)
	coef = zeros(k, block.n)
	for i in eachindex(y)
		g = block.refs[i]
		for c in 1:k
			coef[c, g] += weights[i] * block.interactions[c][i] * y[i]
		end
	end
	for g in 1:block.n
		R = transform[:, :, g]
		coef[:, g] = R * (R' * coef[:, g])
	end
	for i in eachindex(y)
		g = block.refs[i]
		for c in 1:k
			y[i] -= block.interactions[c][i] * coef[c, g]
		end
	end
	return y
end

@testset "block operator identities" begin
	id = [1, 1, 1, 2, 2, 3, 3, 3]
	slope = [0.0, 1.0, 2.0, 1.0, 1.0, -1.0, 0.0, 1.0]
	y = [1.0, 0.5, 2.0, -1.0, 3.0, 2.5, -0.5, 1.5]
	z = [-2.0, 1.0, 0.0, 4.0, 3.0, -1.0, 2.0, 0.5]
	weights = Weights([1.0, 1.5, 2.0, 0.75, 1.25, 1.1, 0.9, 1.3])
	fes_block = [FixedEffect(id), FixedEffect(id, interaction = slope), FixedEffect(id, interaction = 2 .* slope)]
	feM = FixedEffects.AbstractFixedEffectSolver{Float64}(fes_block, weights, Val{:cpu})

	coef = FixedEffects.FixedEffectCoefficients([randn(size(blockcoef)) for blockcoef in feM.x.x])
	Acoef = zeros(length(y))
	mul!(Acoef, feM.m, coef, 1.0, 0.0)
	adj = similar(coef)
	fill!(adj, 0.0)
	mul!(adj, feM.m', z, 1.0, 0.0)
	@test dot(Acoef, z) ≈ sum(dot(a, b) for (a, b) in zip(coef.x, adj.x)) atol = 1e-10

	r1 = copy(y)
	project_block!(r1, feM.m.plan.blocks[1], feM.m.plan.transforms[1], weights)
	r2 = copy(r1)
	project_block!(r2, feM.m.plan.blocks[1], feM.m.plan.transforms[1], weights)
	@test r2 ≈ r1 atol = 1e-10
	rz = copy(z)
	project_block!(rz, feM.m.plan.blocks[1], feM.m.plan.transforms[1], weights)
	@test dot(weights .* r1, z) ≈ dot(weights .* y, rz) atol = 1e-10
	for g in 1:3
		for s in (ones(length(y)), slope, 2 .* slope)
			@test sum(weights[i] * s[i] * r1[i] for i in eachindex(y) if id[i] == g) ≈ 0 atol = 1e-10
		end
	end
end

@testset "rank tolerance with large collinear groups" begin
	n = 200_000
	id = mod1.(1:n, 4)
	slope = randn(n)
	y = randn(n)
	fes_dup = [FixedEffect(id), FixedEffect(id, interaction = slope), FixedEffect(id, interaction = 2 .* slope)]
	plan = FixedEffects.AbsorptionPlan(Float64, fes_dup, uweights(n))
	# slope and 2 * slope are exactly collinear: the rank must be 2 even though the
	# orthogonalization residual of the duplicated column in a 50_000-row group is
	# rounding noise far above eps
	@test plan.ranks[1] == fill(2, 4)
	r_dup = solve_residuals!(copy(y), fes_dup)[1]
	r_ref = solve_residuals!(copy(y), [FixedEffect(id), FixedEffect(id, interaction = slope)])[1]
	@test r_dup ≈ r_ref atol = 1e-8
	coefs = solve_coefficients!(copy(y), fes_dup)[1]
	@test all(maximum(abs, c) < 1e6 for c in coefs)
end

@testset "threaded gather parity" begin
	n = 150_000
	id1 = mod1.(1:n, 100)
	id2 = mod1.(7 .* (1:n) .+ 3, 31)
	xslope = randn(n)
	fes = [FixedEffect(id1), FixedEffect(id1, interaction = xslope), FixedEffect(id2)]
	y = randn(n)
	r_default = solve_residuals!(copy(y), fes)[1]
	FixedEffects._USE_THREADED_GATHER[] = false
	r_serial = solve_residuals!(copy(y), fes)[1]
	FixedEffects._USE_THREADED_GATHER[] = true
	@test r_default ≈ r_serial atol = 1e-8
end
