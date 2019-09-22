##############################################################################
##
## LSMR GPU
##
##############################################################################
using .CuArrays
using .CuArrays.CUDAnative
import .CuArrays: allowscalar

allowscalar(false)

const defaultThreads = 256
##############################################################################
##
## Implement special mean! and deman! (code by Paul Schrimpf)
##
##############################################################################

function mean!(fecoef::CuVector, refs::CuVector, y::CuVector, α::Number, cache::CuVector)
    nthreads = defaultThreads
    nblocks = cld(length(y), nthreads) 
    @cuda threads=nthreads blocks=nblocks mean_kernel!(fecoef, refs, y, α, cache)    
end

function mean_kernel!(fecoef, refs, y, α, cache)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for i = index:stride:length(y)
        CuArrays.CUDAnative.atomic_add!(pointer(fecoef, refs[i]), y[i] * α * cache[i])
    end
end

function demean!(y::CuVector, fecoef::CuVector, refs::CuVector, α::Number, cache::CuVector)
    nthreads = defaultThreads
    nblocks = cld(length(y), nthreads)
    @cuda threads=nthreads blocks=nblocks demean_kernel!(y, fecoef, refs, α, cache)
end

function demean_kernel!(y, fecoef, refs, α, cache)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for i = index:stride:length(y)
    	y[i] += fecoef[refs[i]] * α * cache[i]
    end
end

function scale!(fecoef::CuVector, refs::CuVector, y::CuVector, sqrtw::CuVector)
    nthreads = defaultThreads
    nblocks = cld(length(refs), nthreads) 
	@cuda threads=nthreads blocks=nblocks scale_kernel!(fecoef, refs, y, sqrtw)
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
    nthreads = defaultThreads
    nblocks = cld(length(y), nthreads) 
	@cuda threads=nthreads blocks=nblocks cache_kernel!(y, refs, interaction, fecoef, sqrtw)
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
	n = length(sqrtw)
	scales = [scale!(cuzeros(T, fe.n), fe.refs, fe.interaction, sqrtw) for fe in fes_gpu]
	caches = [cache!(cuzeros(T, n), fe.refs, fe.interaction, scale, sqrtw) for (fe, scale) in zip(fes_gpu, scales)]
	xs = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes_gpu])
	v = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes_gpu])
	h = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes_gpu])
	hbar = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes_gpu])
	u = cuzeros(T, n)
	r = cuzeros(T, n)
	FixedEffectLSMRGPU(FixedEffectLSMR(fes_gpu, scales, caches, xs, v, h, hbar, u, r, sqrtw), zeros(T, n), fes)
end


function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyto!(feM.tmp, r)
    _, iterations, converged = solve_residuals!(feM.tmp, feM.m; kwargs...)
	copyto!(r, feM.tmp)
    r, iterations, converged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
    copyto!(feM.tmp, r)
	xs, iterations, converged = _solve_coefficients!(feM.tmp, feM.m)
	xs = Vector{eltype(r)}[collect(x) for x in xs.x]
    full(normalize!(xs, feM.fes; kwargs...), feM.fes), iterations, converged
end
