##############################################################################
##
## AbstractFixedEffectSolver and the public API
##
## solve_residuals! and solve_coefficients! (defined here) are generic over any
## AbstractFixedEffectSolver. A backend provides:
##   AbstractFixedEffectSolver{T}(fes, weights, ::Type{Val{method}})
##   update_weights!(feM, weights)
##   copy_internal!  (both directions, host <-> solver storage)
##   mul! for its linear map and adjoint (used by lsmr!)
## and may override recover_coefficients.
##
##############################################################################
abstract type AbstractFixedEffectSolver{T} end

"""
`solve_residuals!(y, fes, w; method = :cpu, double_precision = method == :cpu, tol = 1e-8, maxiter = 10000)`

Returns ``y_i - X_i'\\beta`` where ``\\beta = argmin_{b} \\sum_i y_i - X_i'b``, where `X` denotes the matrix of fixed effects `fes`.

### Arguments
* `y` : A `AbstractVector`
* `fes`: A `Vector{<:FixedEffect}`
* `w`: A vector of weights, i.e. `AbstractWeights`
* `method` : A symbol between :cpu (default), :CUDA, or :Metal
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to method == :cpu. GPU backends use Float32 by default; Float32 solves use a looser default tolerance and can be less accurate than CPU Float64 solves.
* `tol` : Tolerance. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.
* `maxiter` : Maximum number of LSMR iterations

### Returns
* `res` :  Residual of the least square problem
* `iterations`: Number of iterations
* `converged`: Did the algorithm converge?

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
solve_residuals!(rand(10), [FixedEffect(p1), FixedEffect(p2)])
```
"""
function solve_residuals!(y::AbstractVector{<: Real}, fes::AbstractVector{<: FixedEffect}, w::AbstractWeights = uweights(eltype(y), length(y));
	method::Symbol = :cpu,
	double_precision::Bool = method == :cpu,
	tol::Real = double_precision ? 1e-8 : 1e-6,
	maxiter::Integer = 10000)
	any((length(fe) != size(y, 1) for fe in fes)) && error("FixedEffects must have the same length as y")
	feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, w, Val{method})
	solve_residuals!(y, feM; maxiter = maxiter, tol = tol)
end

function solve_residuals!(r::AbstractVector{<:Real}, feM::AbstractFixedEffectSolver{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	maxiter >= 0 || throw(ArgumentError("maxiter must be non-negative"))
	# One cannot copy view of Vector (r) on GPU, so first collect the vector
	copy_internal!(feM, :r, r)
	if !(feM.weights isa UnitWeights)
		feM.r .*= sqrt.(feM.weights)
	end
	copyto!(feM.b, feM.r)
	fill!(feM.x, zero(T))
	iter, converged = 0, true
	if length(feM.m.plan.blocks) == 1
		mul!(feM.x, feM.m', feM.b, 1, 0)
	else
		_, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
		iter, converged = ch.mvps, ch.isconverged
	end
	converged || @warn "solve_residuals! did not converge within maxiter LSMR iterations; returned values may be inaccurate." iterations=iter maxiter tol
	mul!(feM.r, feM.m, feM.x, -1, 1)
	if !(feM.weights isa UnitWeights)
		feM.r ./=  sqrt.(feM.weights)
	end
	copy_internal!(r, feM, :r)
	return r, iter, converged
end

# A fallback method for collections of x
# The container for data columns does not have to be a vector
# This allows the use of iterators and tuples for xs in downstream packages
# See https://github.com/FixedEffects/FixedEffects.jl/pull/65
function solve_residuals!(xs, feM::AbstractFixedEffectSolver; progress_bar = true, kwargs...)
    iterations = Int[]
    convergeds = Bool[]
    bar = MiniProgressBar(header = "Demean Variables:", color = Base.info_color(), percentage = false, max = length(xs))
    for (j, x) in enumerate(xs)
    	v0 = time()
        _, iteration, converged = solve_residuals!(x, feM; kwargs...)
        v1 = time()
        # remove progress_bar if estimated time lower than 2sec
	    if progress_bar && (j == 1) && ((v1 - v0) * length(xs) <= 2)
	    	progress_bar = false
	    end
    	if progress_bar
    		bar.current = j
    	    showprogress(stdout, bar)
    	end
        push!(iterations, iteration)
        push!(convergeds, converged)
    end
    if progress_bar
    	end_progress(stdout, bar)
    end
    return xs, iterations, convergeds
end

# Guard: without this, a matrix would fall into the collection method above,
# be iterated element-wise, and recurse until a StackOverflowError.
solve_residuals!(::AbstractMatrix, ::AbstractFixedEffectSolver; kwargs...) =
	throw(ArgumentError("pass the columns, e.g. eachcol(X), rather than a matrix"))


"""
Solve a least square problem for a set of FixedEffects

`solve_coefficients!(y, fes, w; method = :cpu, double_precision = method == :cpu, tol = 1e-8, maxiter = 10000)`

Returns ``\\beta = argmin_{b} \\sum_i w_i(y_i - X_i'b)`` where `X` denotes the matrix of fixed effects `fes`.

### Arguments
* `y` : A `AbstractVector` 
* `fes`: A `Vector{<:FixedEffect}`
* `w`: A vector of weights, i.e. `AbstractWeights`
* `method` : A symbol between :cpu (default), :CUDA, or :Metal
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to method == :cpu. GPU backends use Float32 by default; Float32 solves use a looser default tolerance and can be less accurate than CPU Float64 solves.
* `tol` : Tolerance. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.
* `maxiter` : Maximum number of LSMR iterations


### Returns
* ``\\beta`` : Solution of the least square problem
* `iterations`: Number of iterations
* `converged`: Did the algorithm converge?
Fixed effects are generally not unique. We standardize the solution 
in the following way: the mean of fixed effects within connected components is zero
(except for the first).
This gives the unique solution in the case of two fixed effects.

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)
solve_coefficients!(rand(10), [FixedEffect(p1), FixedEffect(p2)])
```
"""
function solve_coefficients!(y::AbstractVector{<: Number}, fes::AbstractVector{<: FixedEffect}, w::AbstractWeights = uweights(eltype(y), length(y));
		method::Symbol = :cpu,
		double_precision::Bool = method == :cpu,
		tol::Real = double_precision ? 1e-8 : 1e-6,
		maxiter::Integer = 10000)
	any((length(fe) != length(y) for fe in fes))  && error("FixedEffects must have the same length as y")
	feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, w, Val{method})
	solve_coefficients!(y, feM; maxiter = maxiter, tol = tol)
end

function solve_coefficients!(r::AbstractVector, feM::AbstractFixedEffectSolver{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	maxiter >= 0 || throw(ArgumentError("maxiter must be non-negative"))
	# One cannot copy view of Vector (r) on GPU, so first collect the vector
	copy_internal!(feM, :b, r)
	if !(feM.weights isa UnitWeights)
		feM.b .*= sqrt.(feM.weights)
	end
	fill!(feM.x, zero(T))
	_, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	ch.isconverged || @warn "solve_coefficients! did not converge within maxiter LSMR iterations; returned values may be inaccurate." iterations=ch.mvps maxiter tol
	recover_coefficients(feM, eltype(r)), ch.mvps, ch.isconverged
end


function recover_coefficients(feM::AbstractFixedEffectSolver{T}, ::Type{Tout}) where {T, Tout}
	return recover_coefficients(T, feM.m.fes, feM.m.plan, Matrix{T}[Array(x) for x in feM.x.x], Tout)
end

# Transform whitened block coefficients back to one vector per input FixedEffect,
# expanded to observation level. `fes` and `coef_blocks` must live on the CPU.
function recover_coefficients(::Type{T}, fes::Vector{<:FixedEffect}, plan::AbsorptionPlan,
		coef_blocks::Vector{<:Matrix}, ::Type{Tout}) where {T, Tout}
	group_coefs = [zeros(T, fe.n) for fe in fes]
	for (coef_block, block, transform) in zip(coef_blocks, plan.blocks, plan.transforms)
		k = block_width(block)
		β = zeros(T, k)
		@inbounds for g in 1:block.n
			for a in 1:k
				s = zero(T)
				for c in 1:k
					s += transform[a, c, g] * coef_block[c, g]
				end
				β[a] = s
			end
			for (column, term_id) in enumerate(block.input_terms)
				group_coefs[term_id][g] = β[column]
			end
		end
	end
	# Fixed-effect coefficients are generally not unique: within each connected
	# component, a constant can be shifted between the scalar (non-interacted)
	# fixed effects. Pin down a solution by demeaning every scalar fixed effect but
	# the first within each component.
	idx = findall(fe -> isa(fe.interaction, UnitWeights), fes)
	length(idx) >= 2 && rescale!(view(group_coefs, idx), view(fes, idx))
	return Vector{Tout}[Tout.(coef[fe.refs]) for (coef, fe) in zip(group_coefs, fes)]
end

function rescale!(fecoefs::AbstractVector{<: Vector{<: Real}}, fes::AbstractVector{<:FixedEffect})
	labels, ncomponents = components(fes)
	shift = zeros(ncomponents)   # per component, total mean moved to the first fixed effect
	sums = zeros(ncomponents)
	counts = zeros(Int, ncomponents)
	# demean all fixed effects except the first
	for j in length(fecoefs):(-1):2
		fecoef, label = fecoefs[j], labels[j]
		fill!(sums, 0.0)
		fill!(counts, 0)
		for g in eachindex(label)
			c = label[g]
			if c > 0
				sums[c] += fecoef[g]
				counts[c] += 1
			end
		end
		for g in eachindex(label)
			c = label[g]
			if c > 0
				fecoef[g] -= sums[c] / counts[c]
			end
		end
		for c in 1:ncomponents
			if counts[c] > 0
				shift[c] += sums[c] / counts[c]
			end
		end
	end
	# rescale the first fixed effect
	fecoef, label = fecoefs[1], labels[1]
	for g in eachindex(label)
		c = label[g]
		if c > 0
			fecoef[g] += shift[c]
		end
	end
end

# Connected components of the graph linking groups of different fixed effects
# through shared observations, via union-find over the group labels of all
# fixed effects. Returns, for each fixed effect, a vector mapping each group to
# its component id (0 for a group with no observation, which can arise from
# subsetting), and the number of components.
function components(fes::AbstractVector{<:FixedEffect})
	offsets = Vector{Int}(undef, length(fes) + 1)
	offsets[1] = 0
	for (j, fe) in enumerate(fes)
		offsets[j + 1] = offsets[j] + fe.n
	end
	# each observation links its first-effect group to its group in every other effect
	parent = collect(1:offsets[end])
	treesize = ones(Int, offsets[end])
	refs1 = fes[1].refs
	for j in 2:length(fes)
		refsj = fes[j].refs
		offset = offsets[j]
		for i in eachindex(refs1)
			_union!(parent, treesize, Int(refs1[i]), offset + Int(refsj[i]))
		end
	end
	seen = falses(offsets[end])
	for (j, fe) in enumerate(fes)
		offset = offsets[j]
		for r in fe.refs
			seen[offset + Int(r)] = true
		end
	end
	labels = Vector{Int}[zeros(Int, fe.n) for fe in fes]
	component_of_root = zeros(Int, offsets[end])
	ncomponents = 0
	for (j, fe) in enumerate(fes)
		label, offset = labels[j], offsets[j]
		for g in 1:fe.n
			seen[offset + g] || continue
			root = _find!(parent, offset + g)
			c = component_of_root[root]
			if c == 0
				ncomponents += 1
				c = ncomponents
				component_of_root[root] = c
			end
			label[g] = c
		end
	end
	return labels, ncomponents
end

# find with path halving
function _find!(parent::Vector{Int}, i::Int)
	@inbounds while parent[i] != i
		parent[i] = parent[parent[i]]
		i = parent[i]
	end
	return i
end

# union by size
function _union!(parent::Vector{Int}, treesize::Vector{Int}, i::Int, j::Int)
	ri = _find!(parent, i)
	rj = _find!(parent, j)
	ri == rj && return
	if treesize[ri] < treesize[rj]
		ri, rj = rj, ri
	end
	@inbounds begin
		parent[rj] = ri
		treesize[ri] += treesize[rj]
	end
	return
end
