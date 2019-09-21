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

function mean!(fecoef::CuVector, refs::CuVector, y::CuVector, α::Number, cache::CuVector)
    nthreads = 256
    nblocks = div(length(y), nthreads) + 1
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
    nthreads = 256
    nblocks = div(length(y), nthreads) + 1
    @cuda threads=nthreads blocks=nblocks demean_kernel!(y, fecoef, refs, α, cache)
end

function demean_kernel!(y, fecoef, refs, α, cache)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for i = index:stride:length(y)
    	y[i] += fecoef[refs[i]] * α * cache[i]
    end
    return nothing
end

function scale!(fecoef::CuVector, refs::CuVector, y::CuVector, sqrtw::CuVector)
	nthreads = 256
	nblocks = div(length(y), nthreads) + 1
	@cuda threads=nthreads blocks=nblocks scale_kernel!(fecoef, refs, y, sqrtw)
	fecoef .= 1.0 ./ sqrt.(fecoef)
end

function scale_kernel!(fecoef, refs, y, sqrtw)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
	    CuArrays.CUDAnative.atomic_add!(pointer(fecoef, refs[i]), abs2(y[i] * sqrtw[i]))
	end
	return nothing
end

function cache!(y::CuVector, refs::CuVector, interaction::CuVector, fecoef::CuVector, sqrtw::CuVector)
	nthreads = 256
	nblocks = div(length(y), nthreads) + 1
	@cuda threads=nthreads blocks=nblocks cache_kernel!(y, refs, interaction, fecoef, sqrtw)
	return y
end

function cache_kernel!(y, refs, interaction, fecoef, sqrtw)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		y[i] += fecoef[refs[i]] * interaction[i] * sqrtw[i]
	end
	return nothing
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
	tmp::Vector{T} 	# used to convert Abstract{Float64} to Vector{Float32}
	tmp2::CuVector{T} # used to convert Vector{Float32} to CuVector{Float32}
	fes::Vector{<:FixedEffect}
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}})
	fes_gpu = [cu(Float32, fe) for fe in fes]
	sqrtw = cu(Float32, sqrtw)
	n = length(sqrtw)
	scales = [scale!(cuzeros(Float32, fe.n), fe.refs, fe.interaction, sqrtw) for fe in fes_gpu]
	caches = [cache!(cuzeros(Float32, n), fe.refs, fe.interaction, scale, sqrtw) for (fe, scale) in zip(fes_gpu, scales)]
	xs = FixedEffectCoefficients([cuzeros(Float32, fe.n) for fe in fes_gpu])
	v = FixedEffectCoefficients([cuzeros(Float32, fe.n) for fe in fes_gpu])
	h = FixedEffectCoefficients([cuzeros(Float32, fe.n) for fe in fes_gpu])
	hbar = FixedEffectCoefficients([cuzeros(Float32, fe.n) for fe in fes_gpu])
	u = cuzeros(Float32, n)
	FixedEffectLSMRGPU(FixedEffectLSMR(fes_gpu, scales, caches, xs, v, h, hbar, u, sqrtw), zeros(Float32, n), cuzeros(Float32, n), fes)
end


function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyto!(feM.tmp, r)
	copyto!(feM.tmp2, feM.tmp)
	x, iterations, converged = solve_residuals!(feM.tmp2, feM.m; kwargs...)
	copyto!(feM.tmp, x)
	copyto!(r, feM.tmp), iterations, converged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyto!(feM.tmp, r)
	copyto!(feM.tmp2, feM.tmp)
	iterations, converged = _solve_coefficients!(feM.tmp2, feM.m)
	full(normalize!(collect.(feM.m.xs.x), feM.fes; kwargs...), feM.fes), iterations, converged
end


