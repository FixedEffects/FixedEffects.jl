##############################################################################
##
## LSMR GPU
##
##############################################################################
using .CuArrays
using .CuArrays.CUDAnative

.CuArrays.allowscalar(false)

# convert FixedEffects between CPU and GPU
function CuArrays.cu(fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = CuArray(convert(Vector{Float32}, fe.interaction))
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

function Base.collect(fe::FixedEffect{<: CuVector})
	refs = collect(fe.refs)
	interaction = collect(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

# convert FixedEffectCoefficient between CPU and GPU
CuArrays.cu(x::FixedEffectCoefficients) = FixedEffectCoefficients(cu.(x.x))
Base.collect(x::FixedEffectCoefficients{<: CuVector}) = FixedEffectCoefficients(collect.(x.x))

# convert FixedEffectLSMR between CPU and GPU
function CuArrays.cu(m::FixedEffectLSMR)
	FixedEffectLSMR(cu.(m.fes), cu.(m.scales), cu.(m.caches), cu(m.xs), cu(m.v), cu(m.h), cu(m.hbar), cu(m.u), CuArray(convert(Vector{Float32}, m.sqrtw)))
end

##############################################################################
##
## Implement special mean! and deman!
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
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}})
	FixedEffectLSMRGPU(cu(FixedEffectMatrix(fes, sqrtw, Val{:lsmr})))
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	cur = cu(r)
	cur, iterations, converged = solve_residuals!(cur, feM.m; kwargs...)
	copyto!(r, collect(cur)), iterations, converged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	cur = cu(r)
	iterations, converged = _solve_coefficients!(r, feM.m)
	xs = collect.(feM.m.xs.x)
	fes = collect.(feM.m.fes)
	full(normalize!(xs, fes; kwargs...), fes), iterations, converged
end
