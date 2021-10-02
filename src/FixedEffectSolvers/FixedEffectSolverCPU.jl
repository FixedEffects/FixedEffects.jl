##############################################################################
## 
## FixedEffectLinearMap
##
## Model matrix of categorical variables
## mutiplied by diag(1/sqrt(∑w * interaction^2, ..., ∑w * interaction^2) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## eltyp
## size
## mul!
##
##############################################################################
mutable struct FixedEffectLinearMapCPU{T}
	fes::Vector{<:FixedEffect}
	scales::Vector{<:AbstractVector}
	caches::Vector{<:AbstractVector}
	tmp::Vector{Union{Nothing, <:AbstractVector}}
	nthreads::Int
end

function FixedEffectLinearMapCPU{T}(fes::Vector{<:FixedEffect}, ::Type{Val{:cpu}}, nthreads) where {T}
	scales = [zeros(T, fe.n) for fe in fes]
	caches = [zeros(T, length(fes[1].interaction)) for fe in fes]
	fecoefs = [[zeros(T, fe.n) for _ in 1:nthreads] for fe in fes]
	return FixedEffectLinearMapCPU{T}(fes, scales, caches, fecoefs, nthreads)
end

LinearAlgebra.adjoint(fem::FixedEffectLinearMapCPU) = Adjoint(fem)

function Base.size(fem::FixedEffectLinearMapCPU, dim::Integer)
	(dim == 1) ? length(fem.fes[1].refs) : (dim == 2) ? sum(fe.n for fe in fem.fes) : 1
end

Base.eltype(x::FixedEffectLinearMapCPU{T}) where {T} = T


function LinearAlgebra.mul!(fecoefs::FixedEffectCoefficients, 
	Cfem::Adjoint{T, FixedEffectLinearMapCPU{T}},
	y::AbstractVector, α::Number, β::Number) where {T}
	fem = adjoint(Cfem)
	rmul!(fecoefs, β)
	for (fecoef, fe, cache, tmp) in zip(fecoefs.x, fem.fes, fem.caches, fem.tmp)
		gather!(fecoef, fe.refs, α, y, cache, tmp, fem.nthreads)
	end
	return fecoefs
end

function gather!(fecoef::AbstractVector, refs::AbstractVector, α::Number, 
	y::AbstractVector, cache::AbstractVector, tmp::AbstractVector, nthreads::Integer)
	n_each = div(length(y), nthreads)
	Threads.@threads for t in 1:nthreads
		fill!(tmp[t], 0.0)
		gather!(tmp[t], refs, α, y, cache, ((t - 1) * n_each + 1):(t * n_each))
	end
	for x in tmp
		fecoef .+= x
	end
	gather!(fecoef, refs, α, y, cache, (nthreads * n_each + 1):length(y))
end

function gather!(fecoef::AbstractVector, refs::AbstractVector, α::Number, 
	y::AbstractVector, cache::AbstractVector, irange::AbstractRange)
	@inbounds @simd for i in irange
		fecoef[refs[i]] += α * y[i] * cache[i]
	end
end

function LinearAlgebra.mul!(y::AbstractVector, fem::FixedEffectLinearMapCPU, 
			  fecoefs::FixedEffectCoefficients, α::Number, β::Number)
	rmul!(y, β)
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		scatter!(y, α, fecoef, fe.refs, cache, fem.nthreads)
	end
	return y
end

function scatter!(y::AbstractVector, α::Number, fecoef::AbstractVector, 
	refs::AbstractVector, cache::AbstractVector, nthreads::Integer)
	n_each = div(length(y), nthreads)
	Threads.@threads for t in 1:nthreads
		scatter!(y, α, fecoef, refs, cache, ((t - 1) * n_each + 1):(t * n_each))
	end
	scatter!(y, α, fecoef, refs, cache, (nthreads * n_each + 1):length(y))
end

function scatter!(y::AbstractVector, α::Number, fecoef::AbstractVector, 
	refs::AbstractVector, cache::AbstractVector, irange::AbstractRange)
	@inbounds @simd for i in irange
		y[i] += α * fecoef[refs[i]] * cache[i]
	end
end

##############################################################################
##
## Implement AbstractFixedEffectSolver interface
##
##############################################################################

mutable struct FixedEffectSolverCPU{T} <: AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMapCPU{T}
	weights::AbstractVector
	b::AbstractVector{T}
	r::AbstractVector{T}
	x::FixedEffectCoefficients{<: AbstractVector{T}}
	v::FixedEffectCoefficients{<: AbstractVector{T}}
	h::FixedEffectCoefficients{<: AbstractVector{T}}
	hbar::FixedEffectCoefficients{<: AbstractVector{T}}
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:cpu}}, nthreads = Threads.nthreads()) where {T}
	m = FixedEffectLinearMapCPU{T}(fes, Val{:cpu}, nthreads)
	b = zeros(T, length(weights))
	r = zeros(T, length(weights))
	x = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	return update_weights!(FixedEffectSolverCPU(m, weights, b, r, x, v, h, hbar), weights)
end

function update_weights!(feM::FixedEffectSolverCPU, weights::AbstractWeights)
	for (scale, fe) in zip(feM.m.scales, feM.m.fes)
		scale!(scale, fe.refs, fe.interaction, weights)
	end
	for (cache, scale, fe) in zip(feM.m.caches, feM.m.scales, feM.m.fes)
		cache!(cache, fe.refs, fe.interaction, weights, scale)
	end
	feM.weights = weights
	return feM
end


function scale!(scale::AbstractVector, refs::AbstractVector, interaction::AbstractVector, weights::AbstractVector)
	@inbounds @simd for i in eachindex(refs)
		scale[refs[i]] += abs2(interaction[i]) * weights[i]
	end
	# Case of interaction variatble equal to zero in the category (issue #97)
	for i in 1:length(scale)
	    scale[i] = scale[i] > 0 ? (1 / sqrt(scale[i])) : 0.0
	end
end

function cache!(cache::AbstractVector, refs::AbstractVector, interaction::AbstractVector, weights::AbstractVector, scale::AbstractVector)
	@inbounds @simd for i in eachindex(cache)
		cache[i] = interaction[i] * sqrt(weights[i]) * scale[refs[i]]
	end
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectSolverCPU{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.r, r)
	if !(feM.weights isa UnitWeights)
		feM.r .*=  sqrt.(feM.weights)
	end
	copyto!(feM.b, feM.r)
	mul!(feM.x, feM.m', feM.b, 1, 0)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	iter, converged = ch.mvps + 1, ch.isconverged
	mul!(feM.r, feM.m, feM.x, -1, 1)
	if !(feM.weights isa UnitWeights)
		feM.r ./=  sqrt.(feM.weights)
	end
	copyto!(r, feM.r)
	return r, iter, converged
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffects.FixedEffectSolverCPU; progress_bar = true, kwargs...)
    iterations = Int[]
    convergeds = Bool[]
    bar = MiniProgressBar(header = "Demean Variables:", color = Base.info_color(), percentage = false, max = size(X, 2))
    for j in 1:size(X, 2)
    	v0 = time()
        _, iteration, converged = solve_residuals!(view(X, :, j), feM; kwargs...)
        v1 = time()
        # remove progress_bar if estimated time lower than 2sec
	    if progress_bar && (j == 1) && ((v1 - v0) * size(X, 2) <= 2)
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
    return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectSolverCPU{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.b, r)
	if !(feM.weights isa UnitWeights)
		feM.b .*=  sqrt.(feM.weights)
	end
	fill!(feM.x, 0)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	for (x, scale) in zip(feM.x.x, feM.m.scales)
		x .*=  scale
	end
	x = Vector{eltype(r)}[x for x in feM.x.x]
	full(normalize!(x, feM.m.fes; tol = tol, maxiter = maxiter), feM.m.fes), div(ch.mvps, 2), ch.isconverged
end
