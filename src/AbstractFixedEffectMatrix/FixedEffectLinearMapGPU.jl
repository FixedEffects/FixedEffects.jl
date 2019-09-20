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
        r = refs[i]
        CuArrays.CUDAnative.atomic_add!(pointer(fecoef, r), y[i] * α * cache[i])
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
    for i = index:stride:length(y)
    	@inbounds y[i] += fecoef[refs[i]] * α * cache[i]
    end
    return nothing
end

##############################################################################
##
## Implement AbstractRixedEffectMatrix Interface
##
##############################################################################
struct FixedEffectLSMRGPU{T} <: AbstractFixedEffectMatrix{T}
	m::FixedEffectLSMR{T}
	tmp::Vector{T}
	tmp2::CuVector{T}
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}})
	m = FixedEffectMatrix(fes, sqrtw, Val{:lsmr})
	FixedEffectLSMRGPU(
		FixedEffectLSMR(cu.(m.fes), cu.(m.scales), cu.(m.caches), cu(m.xs), cu(m.v), cu(m.h), cu(m.hbar), cu(m.u), CuVector{Float32}(m.sqrtw)),
		zeros(Float32, length(first(m.caches))))
end
function CuArrays.cu(fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = CuVector{Float32}(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end
CuArrays.cu(x::FixedEffectCoefficients) = FixedEffectCoefficients(cu.(x.x))


function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	# views, Float64 to Float32
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
function Base.collect(fe::FixedEffect{<: CuVector})
	refs = collect(fe.refs)
	interaction = collect(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

