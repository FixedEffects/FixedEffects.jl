##############################################################################
## 
## Implement AbstractFixedEffectLinearMap
##
##############################################################################

mutable struct FixedEffectLinearMapCPU{T} <: AbstractFixedEffectLinearMap{T}
	fes::Vector{<:FixedEffect}
	scales::Vector{<:AbstractVector}
	caches::Vector{<:AbstractVector}
	nthreads::Int
end

function FixedEffectLinearMapCPU{T}(fes::Vector{<:FixedEffect}, ::Type{Val{:cpu}}, nthreads) where {T}
	scales = [zeros(T, fe.n) for fe in fes]
	caches = [zeros(T, length(fes[1].interaction)) for fe in fes]
	return FixedEffectLinearMapCPU{T}(fes, scales, caches, nthreads)
end


# multithreaded gather seemds to be slower
function gather!(fecoef::AbstractVector, refs::AbstractVector, α::Number, 
	y::AbstractVector, cache::AbstractVector, nthreads::Integer)
	if α == 1
		@fastmath @inbounds @simd for i in eachindex(y)
			fecoef[refs[i]] += y[i] * cache[i]
		end
	else
		@fastmath @inbounds @simd for i in eachindex(y)
			fecoef[refs[i]] += α * y[i] * cache[i]
		end
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
	# α is actually only 1 or -1 so do special path for them
	if α == 1
		@fastmath @inbounds @simd for i in irange
			y[i] += fecoef[refs[i]] * cache[i]
		end
	elseif α == -1
		@fastmath @inbounds @simd for i in irange
			y[i] -= fecoef[refs[i]] * cache[i]
		end
	else
		@fastmath @inbounds @simd for i in irange
			y[i] += α * fecoef[refs[i]] * cache[i]
		end
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

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:cpu}}, nthreads = nothing) where {T}
	if nthreads === nothing
		nthreads = Threads.nthreads()
	end
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
	@fastmath @inbounds @simd for i in eachindex(refs)
		scale[refs[i]] += abs2(interaction[i]) * weights[i]
	end
	# Case of interaction variatble equal to zero in the category (issue #97)
	T = eltype(scale)
	@fastmath @inbounds @simd for i in eachindex(scale)
	    scale[i] = scale[i] > 0 ? (1 / sqrt(scale[i])) : zero(T)
	end
end

function cache!(cache::AbstractVector, refs::AbstractVector, interaction::AbstractVector, weights::AbstractVector, scale::AbstractVector)
	@fastmath @inbounds @simd for i in eachindex(cache)
		cache[i] = interaction[i] * sqrt(weights[i]) * scale[refs[i]]
	end
end
