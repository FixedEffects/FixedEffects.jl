module FixedEffects

##############################################################################
##
## Dependencies
##
##############################################################################
using LinearAlgebra
using CategoricalArrays
using FillArrays
using StatsBase
using Requires

##############################################################################
##
## Exported methods and types 
##
##############################################################################


export 
group,
FixedEffect,
AbstractFixedEffectSolver,
solve_residuals!,
solve_coefficients!


##############################################################################
##
## Load files
##
##############################################################################
include("lsmr.jl")
include("FixedEffect.jl")
include("AbstractFixedEffectSolver.jl")
include("FixedEffectSolvers/FixedEffectLinearMap.jl")
include("FixedEffectSolvers/FixedEffectSolverCPU.jl")
has_cuarrays() = false
function __init__()
	has_cuarrays() = true
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("FixedEffectSolvers/FixedEffectSolverGPU.jl")
end

end  # module FixedEffects
