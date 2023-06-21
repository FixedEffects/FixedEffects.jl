module FixedEffects

##############################################################################
##
## Dependencies
##
##############################################################################

using Base: @propagate_inbounds
using LinearAlgebra: LinearAlgebra, Adjoint, mul!, rmul!, norm, axpy!
using StatsBase: AbstractWeights, UnitWeights, Weights, uweights
using GroupedArrays: GroupedArray
using Printf: @printf

##############################################################################
##
## Load files
##
##############################################################################
include("utils/lsmr.jl")
include("utils/progressbar.jl")

include("FixedEffect.jl")
include("AbstractFixedEffectSolver.jl")
include("FixedEffectSolvers/LinearMap.jl")
include("FixedEffectSolvers/SolverCPU.jl")


has_CUDA() = false
if !isdefined(Base, :get_extension)
	has_CUDA() = true
	include("../ext/CUDAExt.jl")
end



include("precompile.jl")
_precompile_()

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
