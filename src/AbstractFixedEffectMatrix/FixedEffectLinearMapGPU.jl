##############################################################################
##
## LSMR GPU
##
##############################################################################
using .CuArrays
using .CuArrays.CUDAnative
import .CuArrays: allowscalar

allowscalar(false)

const FloatType = Float64
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
    return nothing
end

function demean!(y::CuVector, fecoef::CuVector, refs::CuVector, α::Number, cache::CuVector)
    nthreads = defaultThreads
    nblocks = cld(length(y), nthreads)
    @cuda threads=nthreads blocks=nblocks demean_kernel!(y, fecoef, refs, α, cache)
end

function demean_kernel!(y, fecoef, refs, α, cache)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(y)
    	@inbounds y[i] += fecoef[refs[i]] * α * cache[i]
    end
    return nothing
end

function _scale!(out::CuVector, fe::FixedEffect, sqrtw::CuVector)
    fill!(out, zero(eltype(out)))
    nthreads = defaultThreads
    nblocks = cld(length(fe.refs), nthreads) 
    @cuda threads=nthreads blocks=nblocks scale_kernel!(out, fe.refs, fe.interaction, sqrtw)
    f(out) = out > 0 ? (1 / sqrt(abs(out))) : zero(out)
    out .= f.(out)
    return out    
end
function scale_kernel!(out, refs, interaction, sqrtw)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for i = index:stride:length(refs)
        CuArrays.CUDAnative.atomic_add!(pointer(out, refs[i]), abs2(interaction[i]*sqrtw[i]))
    end
    return nothing
end

function _cache!(out::CuVector, fe::FixedEffect, scale::CuVector, sqrtw::CuVector)
    nthreads = defaultThreads
    nblocks = cld(length(out), nthreads) 
    @cuda threads=nthreads blocks=nblocks cache_kernel!(out, fe.refs, fe.interaction, scale, sqrtw)
    return out    
end
function cache_kernel!(out, refs, interaction, scale, sqrtw)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for i = index:stride:length(out)
        out[i] = scale[refs[i]] * interaction[i] * sqrtw[i]
    end
    return nothing
end


##############################################################################
##
## Conversion FixedEffect between CPU and GPU
##
##############################################################################
function CuArrays.cu(fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = CuVector{FloatType}(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

function Base.collect(fe::FixedEffect{<: CuVector})
	refs = collect(fe.refs)
    if all(fe.interaction .≈ 1)
        # The check for no interaction in normalize! requires isa(Ones, interaction)
        interaction = Ones{Float64}(length(refs))
    else
	    interaction = collect(fe.interaction)
    end
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

##############################################################################
##
## Implement AbstractRixedEffectMatrix Interface
##
##############################################################################
struct FixedEffectLSMRGPU{T} <: AbstractFixedEffectMatrix{T}
	m::FixedEffectLSMR{T}
	tmp::Vector{T} 	# used to convert views, Float64 to Vector{FloatType}
	tmp2::CuVector{T} # used to convert Vector{FloatType} to CuVector{FloatType}
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}})
    n = length(sqrtw)
    #scales = [_scale!(Vector{FloatType}(undef, fe.n), fe, sqrtw) for fe in fes] 
    #caches = [_cache!(Vector{FloatType}(undef, n), fe, scale, sqrtw) for (fe, scale) in zip(fes, scales)]
    #scales = cu.(scales)
    #caches = cu.(caches)
	fes = cu.(fes)
	sqrtw = CuVector{FloatType}(sqrtw)
    scales = [_scale!(CuVector{FloatType}(undef, fe.n), fe, sqrtw) for fe in fes]
	caches = [_cache!(CuVector{FloatType}(undef, n), fe, scale, sqrtw) for (fe, scale) in zip(fes, scales)]
	xs = FixedEffectCoefficients([CuVector{FloatType}(undef, fe.n) for fe in fes])
	v = FixedEffectCoefficients([CuVector{FloatType}(undef, fe.n) for fe in fes])
	h = FixedEffectCoefficients([CuVector{FloatType}(undef, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([CuVector{FloatType}(undef, fe.n) for fe in fes])
	fill!(v, zero(FloatType))
	fill!(h, zero(FloatType))
	fill!(hbar, zero(FloatType))
	u = CuVector{FloatType}(undef, n)    
	tmp = Vector{FloatType}(undef, n)
    tmp2 = CuVector{FloatType}(undef, n)
	FixedEffectLSMRGPU(FixedEffectLSMR(fes, scales, caches, xs, v, h, hbar, u, sqrtw), tmp, tmp2)
end


function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyto!(feM.tmp, r)
        # CPU to GPU
	copyto!(feM.tmp2, feM.tmp)
	_, iterations, converged = solve_residuals!(feM.tmp2, feM.m; kwargs...)
	copyto!(feM.tmp, feM.tmp2)
	copyto!(r, feM.tmp), iterations, converged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyto!(feM.tmp, r)
	copyto!(feM.tmp2, feM.tmp)
	iterations, converged = _solve_coefficients!(feM.tmp2, feM.m)
	xs = collect.(feM.m.xs.x)
	fes = collect.(feM.m.fes)
	full(normalize!(xs, fes; kwargs...), fes), iterations, converged
end


