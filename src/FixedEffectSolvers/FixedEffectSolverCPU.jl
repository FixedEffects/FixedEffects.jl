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
	colnorm::Vector{<:AbstractVector}
	caches::Vector{<:AbstractVector}
	tmp::Vector{Union{Nothing, <:AbstractVector}}
	nthreads::Int
end

function FixedEffectLinearMapCPU{T}(fes::Vector{<:FixedEffect}, weights::AbstractVector, ::Type{Val{:cpu}}) where {T}
	nthreads = Threads.nthreads()
	colnorm = [_colnorm!(zeros(T, fe.n), fe.refs, fe.interaction, weights) for fe in fes]
	caches = [_cache!(zeros(T, length(weights)), fe.interaction, weights, scale, fe.refs) for (fe, scale) in zip(fes, colnorm)]
	fecoefs = [[zeros(T, fe.n) for _ in 1:nthreads] for fe in fes]
	return FixedEffectLinearMapCPU{T}(fes, colnorm, caches, fecoefs, nthreads)
end

function _colnorm!(fecoef::AbstractVector, refs::AbstractVector, interaction::AbstractVector, weights::AbstractVector)
	@inbounds @simd for i in eachindex(refs)
		fecoef[refs[i]] += abs2(interaction[i]) * weights[i]
	end
	fecoef .= sqrt.(fecoef)
end

function _cache!(y::AbstractVector, interaction::AbstractVector, weights::AbstractVector, fecoef::AbstractVector, refs::AbstractVector)
	@inbounds @simd for i in eachindex(y)
		y[i] = interaction[i] * sqrt(weights[i]) / fecoef[refs[i]]
	end
	return y
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
		_mean!(fecoef, fe.refs, α, y, cache, tmp, fem.nthreads)
	end
	return fecoefs
end

function _mean!(fecoef::AbstractVector, refs::AbstractVector, α::Number, y::AbstractVector, cache::AbstractVector, tmp::AbstractVector, nthreads::Integer)
	n_each = div(length(y), nthreads)
	Threads.@threads for t in 1:nthreads
		fill!(tmp[t], 0.0)
		_mean!(tmp[t], refs, α, y, cache, ((t - 1) * n_each + 1):(t * n_each))
	end
	for x in tmp
		fecoef .+= x
	end
	_mean!(fecoef, refs, α, y, cache, (nthreads * n_each + 1):length(y))
end

function _mean!(fecoef::AbstractVector, refs::AbstractVector, α::Number, y::AbstractVector, cache::AbstractVector, irange::AbstractRange)
	@inbounds @simd for i in irange
		fecoef[refs[i]] += α * y[i] * cache[i]
	end
end

function LinearAlgebra.mul!(y::AbstractVector, fem::FixedEffectLinearMapCPU, 
			  fecoefs::FixedEffectCoefficients, α::Number, β::Number)
	rmul!(y, β)
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		_demean!(y, α, fecoef, fe.refs, cache, fem.nthreads)
	end
	return y
end

function _demean!(y::AbstractVector, α::Number, fecoef::AbstractVector, refs::AbstractVector, cache::AbstractVector, nthreads::Integer)
	n_each = div(length(y), nthreads)
	Threads.@threads for t in 1:nthreads
		_demean!(y, α, fecoef, refs, cache, ((t - 1) * n_each + 1):(t * n_each))
	end
	_demean!(y, α, fecoef, refs, cache, (nthreads * n_each + 1):length(y))
end

function _demean!(y::AbstractVector, α::Number, fecoef::AbstractVector, refs::AbstractVector, cache::AbstractVector, irange::AbstractRange)
	@inbounds @simd for i in irange
		y[i] += α * fecoef[refs[i]] * cache[i]
	end
end

##############################################################################
##
## Implement AbstractFixedEffectSolver interface
##
##############################################################################

struct FixedEffectSolverCPU{T} <: AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMapCPU{T}
	weights::AbstractVector
	b::AbstractVector{T}
	r::AbstractVector{T}
	x::FixedEffectCoefficients{T}
	v::FixedEffectCoefficients{T}
	h::FixedEffectCoefficients{T}
	hbar::FixedEffectCoefficients{T}
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:cpu}}) where {T}
	m = FixedEffectLinearMapCPU{T}(fes, weights, Val{:cpu})
	b = zeros(T, length(weights))
	r = zeros(T, length(weights))
	x = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	return FixedEffectSolverCPU(m, weights, b, r, x, v, h, hbar)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectSolverCPU{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.r, r)
	feM.r .*=  sqrt.(feM.weights)
	fill!(feM.x, 0)
	copyto!(feM.b, feM.r)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	mul!(feM.r, feM.m, feM.x, -1.0, 1.0)
	feM.r ./=  sqrt.(feM.weights)
	copyto!(r, feM.r)
	return r, div(ch.mvps, 2), ch.isconverged
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffects.FixedEffectSolverCPU; kwargs...)
    iterations = Int[]
    convergeds = Bool[]
    for j in 1:size(X, 2)
        _, iteration, converged = solve_residuals!(view(X, :, j), feM; kwargs...)
        push!(iterations, iteration)
        push!(convergeds, converged)
    end
    return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectSolverCPU{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.b, r)
	feM.b .*=  sqrt.(feM.weights)
	fill!(feM.x, 0)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	for (x, scale) in zip(feM.x.x, feM.m.colnorm)
		x ./=  scale
	end
	x = Vector{eltype(r)}[x for x in feM.x.x]
	full(normalize!(x, feM.m.fes; tol = tol, maxiter = maxiter), feM.m.fes), div(ch.mvps, 2), ch.isconverged
end
