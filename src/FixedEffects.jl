
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
include("utils/group.jl")
include("types.jl")
include("FixedEffectMatrix/Common.jl")
include("FixedEffectMatrix/LSMR.jl")
if Base.USE_GPL_LIBS
    include("FixedEffectMatrix/Factorization.jl")
end
include("solve.jl")



end  # module FixedEffectModels