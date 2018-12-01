
module FixedEffects

##############################################################################
##
## Dependencies
##
##############################################################################
import Base: size, copyto!, getindex, length, fill!, eltype, length, view, adjoint
import LinearAlgebra: mul!, rmul!, norm, Matrix, Diagonal, cholesky, cholesky!, Symmetric, Hermitian, rank, dot, eigen, axpy!, svd, I, Adjoint, diag, qr
import LinearAlgebra.BLAS: gemm!
import Statistics: mean
import Distributed: pmap
if Base.USE_GPL_LIBS
    import SparseArrays: SparseMatrixCSC, sparse
end
import CategoricalArrays: CategoricalArray, CategoricalVector, compress, categorical, CategoricalPool, levels, droplevels!

##############################################################################
##
## Exported methods and types 
##
##############################################################################

export group, 
Ones,
FixedEffect,
FixedEffectProblem,
solve_residuals!,
solve_coefficients!


##############################################################################
##
## Load files
##
##############################################################################
include("utils/group.jl")
include("utils/lsmr.jl")
include("utils/Ones.jl")
include("FixedEffectProblem.jl")
include("FixedEffectProblem_LSMR.jl")
if Base.USE_GPL_LIBS
    include("FixedEffectProblem_Factorization.jl")
end
include("solve.jl")



end  # module FixedEffectModels