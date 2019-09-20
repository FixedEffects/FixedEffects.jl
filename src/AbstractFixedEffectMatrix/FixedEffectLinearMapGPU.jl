
##############################################################################
##
## LSMR GPU
##
##############################################################################
using .CuArrays

# convert FixedEffects between CPU and GPU
function CuArrays.CuArray(x::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = CuArray(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

function Base.collect(fe::FixedEffect{<: CuVector})
	refs = collect(fe.refs)
	interaction = collect(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

# convert FixedEffectCoefficient between CPU and GPU
CuArrays.CuArray(x::FixedEffectCoefficients) = FixedEffectCoefficient(CuArray.(x))
Base.collect(x::FixedEffectCoefficients{<: CuVector}) = FixedEffectCoefficient(collect.(x))

# convert FixedEffectLSMR between CPU and GPU
function CuArrays.CuArray(m::FixedEffectLSMR)
	FixedEffectLSMR(CuArray(m.fes), CuArray.(m.scales), CuArray.(m.caches), CuArray(m.xs), CuArray(m.v), CuArray(m.h), CuArray(m.hbar), CuArray(m.u), CuArray(m.sqrtw))
end

##############################################################################
##
## Implement special mean! and deman!
##
##############################################################################

function mean!(fecoef::CuVector, refs::CuVector, y::CuVector, α::Number, cache::CuVector)
    nthreads = 256
    nblocks = div(length(y), nthreads) + 1
    @cuda threads = nthreads blocks = nblocks mean_kernel!(fecoef, refs, y, α, cache)
end

function mean_kernel!(fecoef, refs, y, α, cache)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for i in index:stride:length(y)
        r = refs[i]
        CUDAnative.atomic_add!(pointer(fecoef, r), y[i] * α * cache[i])
    end
    return nothing
end

function demean!(y::CuVector, fecoef::CuVector, refs::CuVector, α::Number, cache::CuVector)
    nthreads = 256
    nblocks = div(length(y), nthreads) + 1
    @cuda threads = nthreads blocks = nblocks demean_kernel!(y, fecoef, refs, α, cache)
end

function demean_kernel!(y, fecoef, refs, α, cache)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for i in index:stride:length(y)
    	y[i] += fecoef[refs[i]] * α * cache[i]
    end
    return nothing
end

##############################################################################
##
## Implement AbstractRixedEffectMatrix Interface
##
##############################################################################
struct FixedEffectLSMRGPU <: AbstractFixedEffectMatrix
	m::FixedEffectLSMR
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}})
	FixedEffectLSMRGPU(CuArray(FixedEffectMatrix(fes, sqrtw, Val{:lsmr})))
end

function solve_residuals!(r::AbstractVector{Float64}, feM::FixedEffectLSMRGPU; kwargs...)
	cur = CuArray(r)
	cur, iterations, converged = solve_residuals!(cur, feM.m; kwargs...)
	copy!(r, cur), iterations, converged
end

function solve_coefficients!(r::AbstractVector{Float64}, feM::FixedEffectLSMRGPU; kwargs...)
	cur = CuArray(r)
	cur .*= feM.sqrtw
	iterations, converged = solve!(feM, r; kwargs...)
	for (x, scale) in zip(feM.xs, feM.scales)
	   x .*=  scale
	end 
	x = collect(feM.xs.x)
	fes = collect(feM.fes)
	full(normalize!(x, fes; kwargs...), fes), iterations, converged
end