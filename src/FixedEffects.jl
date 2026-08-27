module FixedEffects

##############################################################################
##
## Dependencies
##
##############################################################################

using Base: @propagate_inbounds
using Base.Threads: @threads, nthreads
using LinearAlgebra: LinearAlgebra, Adjoint, mul!, rmul!, norm, axpy!
using PrecompileTools: @setup_workload, @compile_workload
using StatsBase: AbstractWeights, UnitWeights, Weights, uweights
using GroupedArrays: GroupedArray, @spawn_for_chunks
using Printf: @printf

##############################################################################
##
## Load files
##
##############################################################################
include("utils/lsmr.jl")
include("utils/progressbar.jl")

include("FixedEffect.jl")
include("AbsorptionPlan.jl")
include("AbstractFixedEffectSolver.jl")
include("FixedEffectCoefficients.jl")
include("AbstractFixedEffectLinearMap.jl")
include("CPU.jl")

##############################################################################
##
## Compatibility shims
##
##############################################################################

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights,
		method::Type{Val{M}}, nthreads::Integer) where {T,M}
	nthreads >= 1 || throw(ArgumentError("nthreads must be positive"))
	return AbstractFixedEffectSolver{T}(fes, weights, method)
end


include("precompile.jl")

##############################################################################
##
## Exported methods and types
##
##############################################################################


export FixedEffect,
AbstractFixedEffectSolver,
solve_residuals!,
solve_coefficients!
end  # module FixedEffects
