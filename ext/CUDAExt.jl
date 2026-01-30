module CUDAExt
using FixedEffects, CUDA
using FixedEffects: FixedEffectCoefficients, AbstractWeights, UnitWeights, LinearAlgebra, Adjoint, mul!, rmul!,  lsmr!, AbstractFixedEffectLinearMap, copy_internal!
CUDA.allowscalar(false)

##############################################################################
##
## Conversion FixedEffect between CPU and GPU
##
##############################################################################

# https://github.com/JuliaGPU/CUDA.jl/issues/142
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
end

function FixedEffectLinearMapCUDA{T}(fes::Vector{<:FixedEffect}) where {T}
	fes = [_cu(T, fe) for fe in fes]
	scales = [CUDA.zeros(T, fe.n) for fe in fes]
	caches = [CUDA.zeros(T, length(fes[1].interaction)) for fe in fes]
	return FixedEffectLinearMapCUDA{T}(fes, scales, caches)
end

function FixedEffects.gather!(fecoef::CuVector, refs::CuVector, α::Number, y::CuVector, cache::CuVector)
	nthreads = 256
	nblocks = cld(length(y), nthreads) 
	@cuda threads=nthreads blocks=nblocks gather_kernel!(fecoef, refs, α, y, cache)    
end

function gather_kernel!(fecoef, refs, α, y, cache)
	index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	i = index
	@inbounds while i <= length(y)
		CUDA.@atomic fecoef[refs[i]] += α * y[i] * cache[i]
		i += stride
	end
end

function FixedEffects.scatter!(y::CuVector, α::Number, fecoef::CuVector, refs::CuVector, cache::CuVector)
	nthreads = 256
	nblocks = cld(length(y), nthreads)
	@cuda threads=nthreads blocks=nblocks scatter_kernel!(y, α, fecoef, refs, cache)
end

function scatter_kernel!(y, α, fecoef, refs, cache)
	index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	i = index
	@inbounds while i <= length(y)
		y[i] += α * fecoef[refs[i]] * cache[i]
		i += stride
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

function FixedEffects.AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:gpu}}, nthreads = nothing) where {T}
	Base.depwarn("The method :gpu is deprecated. Use either :CUDA or :Metal")
	AbstractFixedEffectSolver{T}(fes, weights, Val(:CUDA), nthreads)
end

function FixedEffects.AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:CUDA}}, nthreads = nothing) where {T}
	m = FixedEffectLinearMapCUDA{T}(fes)
	b = CUDA.zeros(T, length(weights))
	r = CUDA.zeros(T, length(weights))
	x = FixedEffectCoefficients([CUDA.zeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([CUDA.zeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([CUDA.zeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([CUDA.zeros(T, fe.n) for fe in fes])
	tmp = zeros(T, length(weights))
	feM = FixedEffectSolverCUDA{T}(m, CUDA.zeros(T, length(weights)), b, r, x, v, h, hbar, tmp, fes)
	FixedEffects.update_weights!(feM, weights)
end

function FixedEffects.update_weights!(feM::FixedEffectSolverCUDA{T}, weights::AbstractWeights) where {T}
	copyto!(feM.weights, _cu(T, weights))
	for (scale, fe) in zip(feM.m.scales, feM.m.fes)
		scale!(scale, fe.refs, fe.interaction, feM.weights)
	end
	for (cache, scale, fe) in zip(feM.m.caches, feM.m.scales, feM.m.fes)
		cache!(cache, fe.refs, fe.interaction, feM.weights, scale)
	end	
	return feM
end

function scale!(scale::CuVector, refs::CuVector, interaction::CuVector, weights::CuVector)
	nthreads = 256
	nblocks = cld(length(refs), nthreads) 
    fill!(scale, 0)
	@cuda threads=nthreads blocks=nblocks scale_kernel!(scale, refs, interaction, weights)
	map!(x -> x > 0 ? 1 / sqrt(x) : zero(eltype(scale)), scale, scale)
end

function scale_kernel!(scale, refs, interaction, weights)
	index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	i = index
	@inbounds while i <= length(interaction)
		CUDA.@atomic scale[refs[i]] += abs2(interaction[i]) * weights[i]
		i += stride
	end
end

function cache!(cache::CuVector, refs::CuVector, interaction::CuVector, weights::CuVector, scale::CuVector)
	nthreads = 256
	nblocks = cld(length(cache), nthreads) 
	@cuda threads=nthreads blocks=nblocks cache!_kernel!(cache, refs, interaction, weights, scale)
end

function cache!_kernel!(cache, refs, interaction, weights, scale)
	index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	i = index
	@inbounds while i <= length(cache)
		cache[i] = interaction[i] * sqrt(weights[i]) * scale[refs[i]]
		i += stride
	end
end

function FixedEffects.copy_internal!(feM::FixedEffectSolverCUDA, field::Symbol, r::AbstractVector)
	copyto!(feM.tmp, r)
	copyto!(getfield(feM, field), feM.tmp)
end

function FixedEffects.copy_internal!(r::AbstractVector, feM::FixedEffectSolverCUDA, field::Symbol)
	copyto!(feM.tmp, getfield(feM, field))
	copyto!(r, feM.tmp)
end


end
