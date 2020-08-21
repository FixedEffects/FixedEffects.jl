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
using ProgressMeter



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


has_CUDA() = false
function __init__()
	    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
	    if CUDA.functional()
		    has_CUDA() = true
	    	include("FixedEffectSolvers/FixedEffectSolverGPU.jl")
	    end
	end
end


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


end  # module FixedEffects
