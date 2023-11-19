module MetalExt
using FixedEffects, Metal
using FixedEffects: FixedEffectCoefficients, AbstractWeights, UnitWeights, LinearAlgebra, Adjoint, mul!, rmul!, lsmr!, AbstractFixedEffectLinearMap
Metal.allowscalar(false)

##############################################################################
##
## Conversion FixedEffect between CPU and Metal
##
##############################################################################

function _mtl(T::Type, fe::FixedEffect)
	refs = MtlArray(fe.refs)
	interaction = _mtl(T, fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end
_mtl(T::Type, w::UnitWeights) = Metal.ones(T, length(w))
_mtl(T::Type, w::AbstractVector) = MtlVector{T}(convert(Vector{T}, w))

##############################################################################
##
## FixedEffectLinearMap on the Metal (code by Paul Schrimpf)
##
## Model matrix of categorical variables
## mutiplied by diag(1/sqrt(∑w * interaction^2, ..., ∑w * interaction^2) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## eltype
## size
## mul!
##
##############################################################################

mutable struct FixedEffectLinearMapMetal{T} <: AbstractFixedEffectLinearMap{T}
	fes::Vector{<:FixedEffect}
	scales::Vector{<:AbstractVector}
	caches::Vector{<:AbstractVector}
	nthreads::Int
end

function FixedEffectLinearMapMetal{T}(fes::Vector{<:FixedEffect}, nthreads) where {T}
	fes = [_mtl(T, fe) for fe in fes]
	scales = [Metal.zeros(T, fe.n) for fe in fes]
	caches = [Metal.zeros(T, length(fes[1].interaction)) for fe in fes]
	return FixedEffectLinearMapMetal{T}(fes, scales, caches, nthreads)
end

function FixedEffects.gather!(fecoef::MtlVector, refs::MtlVector, α::Number, y::MtlVector, cache::MtlVector, nthreads::Integer)
	nblocks = cld(length(y), nthreads) 
	Metal.@sync @metal threads=nthreads groups=nblocks gather_kernel!(fecoef, refs, α, y, cache)    
end

function gather_kernel!(fecoef, refs, α, y, cache)
	i = thread_position_in_grid_1d()
	if i <= length(refs)
		Metal.atomic_fetch_add_explicit(pointer(fecoef, refs[i]), α * y[i] * cache[i])
	end
	return nothing
end

function FixedEffects.scatter!(y::MtlVector, α::Number, fecoef::MtlVector, refs::MtlVector, cache::MtlVector, nthreads::Integer)
	nblocks = cld(length(y), nthreads)
	Metal.@sync @metal threads=nthreads groups=nblocks scatter_kernel!(y, α, fecoef, refs, cache)
end

function scatter_kernel!(y, α, fecoef, refs, cache)
	i = thread_position_in_grid_1d()
	if i <= length(y)
		y[i] += α * fecoef[refs[i]] * cache[i]
	end
	return nothing
end



##############################################################################
##
## Implement AbstractFixedEffectSolver interface
##
##############################################################################

mutable struct FixedEffectSolverMetal{T} <: FixedEffects.AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMapMetal{T}
	weights::MtlVector{T}
	b::MtlVector{T}
	r::MtlVector{T}
	x::FixedEffectCoefficients{<: AbstractVector{T}}
	v::FixedEffectCoefficients{<: AbstractVector{T}}
	h::FixedEffectCoefficients{<: AbstractVector{T}}
	hbar::FixedEffectCoefficients{<: AbstractVector{T}}
	tmp::Vector{T} # used to convert AbstractVector to Vector{T}
	fes::Vector{<:FixedEffect}
end
	
function FixedEffects.AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:Metal}}, nthreads = 256) where {T}
	m = FixedEffectLinearMapMetal{T}(fes, nthreads)
	b = Metal.zeros(T, length(weights))
	r = Metal.zeros(T, length(weights))
	x = FixedEffectCoefficients([Metal.zeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([Metal.zeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([Metal.zeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([Metal.zeros(T, fe.n) for fe in fes])
	tmp = zeros(T, length(weights))
	feM = FixedEffectSolverMetal{T}(m, Metal.zeros(T, length(weights)), b, r, x, v, h, hbar, tmp, fes)
	FixedEffects.update_weights!(feM, weights)
end


function FixedEffects.update_weights!(feM::FixedEffectSolverMetal{T}, weights::AbstractWeights) where {T}
	copyto!(feM.weights, _mtl(T, weights))
	for (scale, fe) in zip(feM.m.scales, feM.m.fes)
		scale!(scale, fe.refs, fe.interaction, feM.weights, feM.m.nthreads)
	end
	for (cache, scale, fe) in zip(feM.m.caches, feM.m.scales, feM.m.fes)
		cache!(cache, fe.refs, fe.interaction, feM.weights, scale, feM.m.nthreads)
	end	
	return feM
end

function scale!(scale::MtlVector, refs::MtlVector, interaction::MtlVector, weights::MtlVector, nthreads::Integer)
	nblocks = cld(length(refs), nthreads) 
    fill!(scale, 0.0)
	Metal.@sync @metal threads=nthreads groups=nblocks scale_kernel!(scale, refs, interaction, weights)
	Metal.@sync @metal threads=nthreads groups=nblocks inv_kernel!(scale)
end

function scale_kernel!(scale, refs, interaction, weights)
	i = thread_position_in_grid_1d()
	if i <= length(refs)
		Metal.atomic_fetch_add_explicit(pointer(scale, refs[i]), interaction[i]^2 * weights[i])
	end
	return nothing
end

function inv_kernel!(scale)
	i = thread_position_in_grid_1d()
	if i <= length(scale)
		scale[i] = (scale[i] > 0) ? (1 / sqrt(scale[i])) : 0.0
	end
	return nothing
end

function cache!(cache::MtlVector, refs::MtlVector, interaction::MtlVector, weights::MtlVector, scale::MtlVector, nthreads::Integer)
	nblocks = cld(length(cache), nthreads) 
	Metal.@sync @metal threads=nthreads groups=nblocks cache!_kernel!(cache, refs, interaction, weights, scale)
end

function cache!_kernel!(cache, refs, interaction, weights, scale)
	i = thread_position_in_grid_1d()
	if i <= length(cache)
		cache[i] = interaction[i] * sqrt(weights[i]) * scale[refs[i]]
	end
	return nothing
end


end
