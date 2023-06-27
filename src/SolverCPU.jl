##############################################################################
## 
## Implement AbstractFixedEffectLinearMap
##
##############################################################################

mutable struct FixedEffectLinearMapCPU{T} <: AbstractFixedEffectLinearMap{T}
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

works_with_view(x::FixedEffectSolverCPU) = true

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
        fill!(scale, 0)
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




