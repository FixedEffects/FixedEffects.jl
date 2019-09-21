
module FixedEffects

##############################################################################
##
## Dependencies
##
##############################################################################
import Base: size, copyto!, getindex, length, fill!, eltype, length, view, adjoint, show, ismissing
import LinearAlgebra: mul!, rmul!, norm, Matrix, Diagonal, cholesky, cholesky!, Symmetric, Hermitian, rank, dot, eigen, axpy!, svd, I, Adjoint, adjoint, diag, qr
if Base.USE_GPL_LIBS
    import SparseArrays: SparseMatrixCSC, sparse
end
import Distributed: pmap
import CategoricalArrays: CategoricalArray, CategoricalVector, compress, categorical, CategoricalPool, levels, droplevels!
using FillArrays
using Reexport
using Requires
@reexport using StatsBase
##############################################################################
##
## Exported methods and types 
##
##############################################################################

export 
group,
FixedEffect,
FixedEffectMatrix,
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
Base.USE_GPL_LIBS && include("AbstractFixedEffectMatrix/FixedEffectCSC.jl")

## @require within __init___ breaks Revise.jl
#using CuArrays  
#include("AbstractFixedEffectMatrix/FixedEffectLinearMapGPU.jl")

function __init__()
	@require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("AbstractFixedEffectMatrix/FixedEffectLinearMapGPU.jl")
end


end  # module FixedEffectModels
