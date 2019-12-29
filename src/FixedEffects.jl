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
using CUDAapi
if has_cuda()
	try
		using CuArrays
		@eval has_cuarrays() = true
	catch
		@info "CUDA is installed, but CuArrays.jl fails to load"
		@eval has_cuarrays() = false
	end
else
	has_cuarrays() = false
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
if has_cuarrays()
	include("FixedEffectSolvers/FixedEffectSolverGPU.jl")
end

end  # module FixedEffects
