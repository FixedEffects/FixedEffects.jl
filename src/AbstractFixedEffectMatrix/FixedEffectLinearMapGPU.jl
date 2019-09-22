##############################################################################
##
## LSMR GPU
##
##############################################################################
using .CuArrays
using .CuArrays.CUDAnative
import .CuArrays: allowscalar

allowscalar(false)

#const FloatType = Float64
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
    @inbounds for i = index:stride:length(y)
    	y[i] += fecoef[refs[i]] * α * cache[i]
    end
    return nothing
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
	return nothing
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
	return nothing
end

##############################################################################
##
## Conversion FixedEffect between CPU and GPU
##
##############################################################################

# function Base.collect(fe::FixedEffect{<: CuVector})
# 	refs = collect(fe.refs)
#     if all(fe.interaction .≈ 1)
#         # The check for no interaction in normalize! requires isa(Ones, interaction)
#         interaction = Ones{Float64}(length(refs))
#     else
# 	    interaction = collect(fe.interaction)
#     end
# end


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
    
function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector{T}, ::Type{Val{:lsmr_gpu}}) where {T}
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
	FixedEffectLSMRGPU(FixedEffectLSMR(fes_gpu, scales, caches, xs, v, h, hbar, u, sqrtw), zeros(T, n), cuzeros(T, n), fes)
end

function copyresid!(r::AbstractVector, feM::FixedEffectLSMRGPU)
    copyto!(feM.tmp, feM.tmp2)
    copyto!(r, feM.tmp)
end


function copyresid!(feM::FixedEffectLSMRGPU, r::AbstractVector)
    copyto!(feM.tmp, r)
    copyto!(feM.tmp2, feM.tmp)   
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyresid!(feM,r) 
    _, iterations, converged = solve_residuals!(feM.tmp2, feM.m; kwargs...)
	copyresid!(r,feM) 
    r, iterations, converged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
    copyresid!(feM, r)
	fecoefs, iterations, converged = _solve_coefficients!(feM.tmp2, feM.m)
    (full(normalize!(collect.(fecoefs.x), feM.fes; kwargs...), feM.fes),
     iterations, converged)
end

