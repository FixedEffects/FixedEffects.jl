module CUDAExt
using FixedEffects, CUDA
using FixedEffects: FixedEffectCoefficients, AbstractWeights, UnitWeights, LinearAlgebra, Adjoint, mul!, rmul!,  lsmr!, AbstractFixedEffectLinearMap
CUDA.allowscalar(false)

##############################################################################
##
## Conversion FixedEffect between CPU and GPU
##
##############################################################################

# https://github.com/JuliaGPU/CUDA.jl/issues/142
_cuzeros(T::Type, n::Integer) = fill!(CuVector{T}(undef, n), zero(T))
function _cu(T::Type, fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = _cu(T, fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end
_cu(T::Type, w::UnitWeights) = fill!(CuVector{T}(undef, length(w)), w[1])
_cu(T::Type, w::AbstractVector) = CuVector{T}(convert(Vector{T}, w))

##############################################################################
##
## FixedEffectLinearMap on the GPU (code by Paul Schrimpf)
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

mutable struct FixedEffectLinearMapCUDA{T} <: AbstractFixedEffectLinearMap{T}
	fes::Vector{<:FixedEffect}
	scales::Vector{<:AbstractVector}
	caches::Vector{<:AbstractVector}
	nthreads::Int
end

function FixedEffectLinearMapCUDA{T}(fes::Vector{<:FixedEffect}, ::Type{Val{:CUDA}}, nthreads) where {T}
	fes = [_cu(T, fe) for fe in fes]
	scales = [_cuzeros(T, fe.n) for fe in fes]
	caches = [_cuzeros(T, length(fes[1].interaction)) for fe in fes]
	return FixedEffectLinearMapCUDA{T}(fes, scales, caches, nthreads)
end


function FixedEffects.gather!(fecoef::CuVector, refs::CuVector, α::Number, y::CuVector, cache::CuVector, nthreads::Integer)
	nblocks = cld(length(y), nthreads) 
	@cuda threads=nthreads blocks=nblocks gather_kernel!(fecoef, refs, α, y, cache)    
end

function gather_kernel!(fecoef, refs, α, y, cache)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		CUDA.atomic_add!(pointer(fecoef, refs[i]), α * y[i] * cache[i])
	end
end

function FixedEffects.scatter!(y::CuVector, α::Number, fecoef::CuVector, refs::CuVector, cache::CuVector, nthreads::Integer)
	nblocks = cld(length(y), nthreads)
	@cuda threads=nthreads blocks=nblocks scatter_kernel!(y, α, fecoef, refs, cache)
end

function scatter_kernel!(y, α, fecoef, refs, cache)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		y[i] += α * fecoef[refs[i]] * cache[i]
	end
end

##############################################################################
##
## Implement AbstractFixedEffectSolver interface
##
##############################################################################

mutable struct FixedEffectSolverCUDA{T} <: FixedEffects.AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMapCUDA{T}
	weights::CuVector{T}
	b::CuVector{T}
	r::CuVector{T}
	x::FixedEffectCoefficients{<: AbstractVector{T}}
	v::FixedEffectCoefficients{<: AbstractVector{T}}
	h::FixedEffectCoefficients{<: AbstractVector{T}}
	hbar::FixedEffectCoefficients{<: AbstractVector{T}}
	tmp::Vector{T} # used to convert AbstractVector to Vector{T}
	fes::Vector{<:FixedEffect}
end
	

FixedEffects.works_with_view(x::FixedEffectSolverCUDA) = false


function FixedEffects.AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:CUDA}}, nthreads = 256) where {T}
	m = FixedEffectLinearMapCUDA{T}(fes, Val{:CUDA}, nthreads)
	b = _cuzeros(T, length(weights))
	r = _cuzeros(T, length(weights))
	x = FixedEffectCoefficients([_cuzeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([_cuzeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([_cuzeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([_cuzeros(T, fe.n) for fe in fes])
	tmp = zeros(T, length(weights))
	FixedEffects.update_weights!(FixedEffectSolverCUDA{T}(m, _cuzeros(T, length(weights)), b, r, x, v, h, hbar, tmp, fes), weights)
end

function FixedEffects.update_weights!(feM::FixedEffectSolverCUDA{T}, weights::AbstractWeights) where {T}
	weights = _cu(T, weights)
	nthreads = feM.m.nthreads
	for (scale, fe) in zip(feM.m.scales, feM.m.fes)
		scale!(scale, fe.refs, fe.interaction, weights, nthreads)
	end
	for (cache, scale, fe) in zip(feM.m.caches, feM.m.scales, feM.m.fes)
		cache!(cache, fe.refs, fe.interaction, weights, scale, nthreads)
	end	
	feM.weights = weights
	return feM
end

function scale!(scale::CuVector, refs::CuVector, interaction::CuVector, weights::CuVector, nthreads::Integer)
	nblocks = cld(length(refs), nthreads) 
        fill!(scale, 0)
	@cuda threads=nthreads blocks=nblocks scale_kernel!(scale, refs, interaction, weights)
	@cuda threads=nthreads blocks=nblocks inv_kernel!(scale)
end

function scale_kernel!(scale, refs, interaction, weights)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(interaction)
		CUDA.atomic_add!(pointer(scale, refs[i]), abs2(interaction[i]) * weights[i])
	end
end

function inv_kernel!(scale)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(scale)
		scale[i] = (scale[i] > 0) ? (1 / sqrt(scale[i])) : 0.0
	end
end

function cache!(cache::CuVector, refs::CuVector, interaction::CuVector, weights::CuVector, scale::CuVector, nthreads::Integer)
	nblocks = cld(length(cache), nthreads) 
	@cuda threads=nthreads blocks=nblocks cache!_kernel!(cache, refs, interaction, weights, scale)
end

function cache!_kernel!(cache, refs, interaction, weights, scale)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(cache)
		cache[i] = interaction[i] * sqrt(weights[i]) * scale[refs[i]]
	end
end



end
