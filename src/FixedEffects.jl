
module FixedEffects

##############################################################################
##
## Dependencies
##
##############################################################################
import Base: size, copyto!, getindex, length, fill!, eltype, length, view, adjoint, show, ismissing
import LinearAlgebra: mul!, rmul!, norm, Matrix, Diagonal, cholesky, cholesky!, Symmetric, Hermitian, rank, dot, eigen, axpy!, svd, I, Adjoint, adjoint, diag, qr
import Distributed: pmap
import CategoricalArrays: CategoricalArray, CategoricalVector, compress, categorical, CategoricalPool, levels, droplevels!
using FillArrays
using Reexport
using CUDAapi
if has_cuda()
	try
		using CuArrays
		@eval has_cuarrays() = true
	catch
		@info "CUDA is installed, but CuArrays.jl fails to load"
		@eval has_cuarrays() = false
	end
end
@reexport using StatsBase # used for Weights
##############################################################################
##
## Exported methods and types 
##
##############################################################################

export 
group,
FixedEffect,
AbstractFixedEffectMatrix,
solve_residuals!,
solve_coefficients!


##############################################################################
##
## Load files
##
##############################################################################
include("utils/lsmr.jl")
include("FixedEffect.jl")
include("solve.jl")
include("AbstractFixedEffectMatrix/FixedEffectLinearMap.jl")
include("AbstractFixedEffectMatrix/FixedEffectLinearMapParallel.jl")

if has_cuarrays()
	include("AbstractFixedEffectMatrix/FixedEffectLinearMapGPU.jl")
end


end  # module FixedEffectModels
