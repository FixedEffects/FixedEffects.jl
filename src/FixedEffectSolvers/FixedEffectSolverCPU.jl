##############################################################################
##
## FixedEffectLinearMap on the CPU
##
##############################################################################

function FixedEffectLinearMap{T}(fes::Vector{<:FixedEffect}, weights::AbstractVector, ::Type{Val{:cpu}}) where {T}
	sqrtw = convert(AbstractVector{T}, sqrt.(values(weights)))
	colnorm = [_colnorm!(zeros(T, fe.n), fe.refs, fe.interaction, sqrtw) for fe in fes]
	caches = [_cache!(zeros(T, length(sqrtw)), fe.interaction, sqrtw, scale, fe.refs) for (fe, scale) in zip(fes, colnorm)]
	return FixedEffectLinearMap{T}(fes, sqrtw, colnorm, caches)
end

function _colnorm!(fecoef::AbstractVector, refs::AbstractVector, interaction::AbstractVector, sqrtw::AbstractVector)
	@inbounds @simd for i in eachindex(refs)
		fecoef[refs[i]] += abs2(interaction[i] * sqrtw[i])
	end
	fecoef .= sqrt.(fecoef)
end

function _cache!(y::AbstractVector, interaction::AbstractVector, sqrtw::AbstractVector, fecoef::AbstractVector, refs::AbstractVector)
	@inbounds @simd for i in eachindex(y)
		y[i] = interaction[i] * sqrtw[i] / fecoef[refs[i]]
	end
	return y
end

function demean!(y::AbstractVector, α::Number, fecoef::AbstractVector, refs::AbstractVector, cache::AbstractVector)
	@simd ivdep for i in eachindex(y)
		@inbounds y[i] += α * fecoef[refs[i]] * cache[i]
	end
end

function mean!(fecoef::AbstractVector, refs::AbstractVector, α::Number, y::AbstractVector, cache::AbstractVector)
	@simd ivdep for i in eachindex(y)
		@inbounds fecoef[refs[i]] += α * y[i] * cache[i]
	end
end

##############################################################################
##
## Implement AbstractFixedEffectSolver interface
##
##############################################################################

struct FixedEffectSolver{T} <: AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMap{T}
	b::AbstractVector{T}
	r::AbstractVector{T}
	x::FixedEffectCoefficients{T}
	v::FixedEffectCoefficients{T}
	h::FixedEffectCoefficients{T}
	hbar::FixedEffectCoefficients{T}
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:cpu}}) where {T}
	m = FixedEffectLinearMap{T}(fes, weights, Val{:cpu})
	b = zeros(T, length(weights))
	r = zeros(T, length(weights))
	x = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	return FixedEffectSolver(m, b, r, x, v, h, hbar)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectSolver{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.r, r)
	feM.r .*=  feM.m.sqrtw
	fill!(feM.x, 0)
	copyto!(feM.b, feM.r)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	mul!(feM.r, feM.m, feM.x, -1.0, 1.0)
	feM.r ./=  feM.m.sqrtw
	copyto!(r, feM.r)
	return r, div(ch.mvps, 2), ch.isconverged
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectSolver; kwargs...)
	iterations = Int[]
	convergeds = Bool[]
	feMs = [feM, (deepcopy(feM) for _ in 2:Threads.nthreads())...]
	Threads.@threads for j in 1:size(X, 2)
		_, iteration, converged = solve_residuals!(view(X, :, j), feMs[Threads.threadid()]; kwargs...)
		push!(iterations, iteration)
		push!(convergeds, converged)
	end
	return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectSolver{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.b, r)
	feM.b .*=  feM.m.sqrtw
	fill!(feM.x, 0)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	for (x, scale) in zip(feM.x.x, feM.m.colnorm)
		x ./=  scale
	end
	x = Vector{eltype(r)}[x for x in feM.x.x]
	full(normalize!(x, feM.m.fes; tol = tol, maxiter = maxiter), feM.m.fes), div(ch.mvps, 2), ch.isconverged
end
