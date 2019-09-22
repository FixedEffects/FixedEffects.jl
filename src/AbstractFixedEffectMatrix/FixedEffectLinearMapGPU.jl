##############################################################################
##
## LSMR GPU
##
##############################################################################
using .CuArrays
using .CuArrays.CUDAnative
import .CuArrays: allowscalar
allowscalar(false)

##############################################################################
##
## Implement special mean! and deman! (code by Paul Schrimpf)
##
##############################################################################
const N_THREADS = 256

function mean!(fecoef::CuVector, refs::CuVector, y::CuVector, α::Number, cache::CuVector)
	nblocks = cld(length(y), N_THREADS) 
	@cuda threads=N_THREADS blocks=nblocks mean_kernel!(fecoef, refs, y, α, cache)    
end

function mean_kernel!(fecoef, refs, y, α, cache)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		CuArrays.CUDAnative.atomic_add!(pointer(fecoef, refs[i]), y[i] * α * cache[i])
	end
end

function demean!(y::CuVector, fecoef::CuVector, refs::CuVector, α::Number, cache::CuVector)
	nblocks = cld(length(y), N_THREADS)
	@cuda threads=N_THREADS blocks=nblocks demean_kernel!(y, fecoef, refs, α, cache)
end

function demean_kernel!(y, fecoef, refs, α, cache)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		y[i] += fecoef[refs[i]] * α * cache[i]
	end
end

function scale!(fecoef::CuVector, refs::CuVector, y::CuVector, sqrtw::CuVector)
	nblocks = cld(length(refs), N_THREADS) 
	@cuda threads=N_THREADS blocks=nblocks scale_kernel!(fecoef, refs, y, sqrtw)
	fecoef .= one(eltype(fecoef)) ./ sqrt.(fecoef)
end

function scale_kernel!(fecoef, refs, y, sqrtw)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		CuArrays.CUDAnative.atomic_add!(pointer(fecoef, refs[i]), abs2(y[i] * sqrtw[i]))
	end
end

function cache!(y::CuVector, refs::CuVector, interaction::CuVector, fecoef::CuVector, sqrtw::CuVector)
	nblocks = cld(length(y), N_THREADS) 
	@cuda threads=N_THREADS blocks=nblocks cache_kernel!(y, refs, interaction, fecoef, sqrtw)
	return y
end

function cache_kernel!(y, refs, interaction, fecoef, sqrtw)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		y[i] += fecoef[refs[i]] * interaction[i] * sqrtw[i]
	end
end

##############################################################################
##
## Conversion FixedEffect between CPU and GPU
##
##############################################################################

# https://github.com/JuliaGPU/CuArrays.jl/issues/306
cuzeros(T::Type, n::Integer) = fill!(CuVector{T}(undef, n), zero(T))
cuones(T::Type, n::Integer) = fill!(CuVector{T}(undef, n), one(T))

function CuArrays.cu(T::Type, fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = cu(T, fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end
CuArrays.cu(T::Type, w::Ones) = cuones(T, length(w))
CuArrays.cu(T::Type, w::AbstractVector) = CuVector{T}(w)

##############################################################################
##
## Implement AbstractRixedEffectMatrix Interface
##
## GC accounts for large part of timing so important to have temporary arrays
##
##############################################################################

struct FixedEffectLSMRGPU{T} <: AbstractFixedEffectMatrix{T}
	m::FixedEffectLSMR{T}
	tmp::Vector{T} # used to convert AbstractVector to Vector{T}
	fes::Vector{<:FixedEffect}
end
	
function AbstractFixedEffectMatrix{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}}) where {T}
	fes_gpu = [cu(T, fe) for fe in fes]
	sqrtw = cu(T, sqrtw)
	scales = [scale!(cuzeros(T, fe.n), fe.refs, fe.interaction, sqrtw) for fe in fes_gpu]
	caches = [cache!(cuzeros(T, length(sqrtw)), fe.refs, fe.interaction, scale, sqrtw) for (fe, scale) in zip(fes_gpu, scales)]
	xs = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes_gpu])
	v = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes_gpu])
	h = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes_gpu])
	hbar = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes_gpu])
	u = cuzeros(T, length(sqrtw))
	r = cuzeros(T, length(sqrtw))
	FixedEffectLSMRGPU(FixedEffectLSMR(fes_gpu, scales, caches, xs, v, h, hbar, u, r, sqrtw), zeros(T, length(sqrtw)), fes)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyto!(feM.tmp, r)
	_, iterations, converged = solve_residuals!(feM.tmp, feM.m; kwargs...)
	copyto!(r, feM.tmp)
	r, iterations, converged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyto!(feM.tmp, r)
	copyto!(feM.m.r, feM.tmp)
	feM.m.r .*= feM.m.sqrtw
	iterations, converged = solve!(feM.m, feM.m.r; kwargs...)
	for (x, scale) in zip(feM.m.xs.x, feM.m.scales)
		x .*=  scale
	end
	xs = Vector{eltype(r)}[collect(x) for x in feM.m.xs.x]
	full(normalize!(xs, feM.fes; kwargs...), feM.fes), iterations, converged
end