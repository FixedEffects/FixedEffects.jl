##############################################################################
##
## Least Squares via Cholesky Factorization
##
##############################################################################

struct CholeskyFixedEffectMatrix{T, V, N} <: AbstractFixedEffectMatrix
    fes::Vector{<:FixedEffect}
    m::SparseMatrixCSC{V, N}
    cholm::T
    x::Vector{V}
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:cholesky}})
    m = sparse(fes, sqrtw)
    cholm = cholesky(Symmetric(m' * m))
    total_len = sum(length(unique(fe.refs)) for fe in fes)
    CholeskyFixedEffectMatrix(fes, m, cholm, zeros(total_len))
end

function solve!(fep::CholeskyFixedEffectMatrix, r::AbstractVector; kwargs...)
    fep.cholm \ mul!(fep.x, fep.m', r)
end

##############################################################################
##
## Least Squares via QR Factorization
##
##############################################################################

struct QRFixedEffectMatrix{T, V, N} <: AbstractFixedEffectMatrix
    fes::Vector{<:FixedEffect}
    m::SparseMatrixCSC{V, N}
    qrm::T
    b::Vector{V}
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:qr}})
    m = sparse(fes, sqrtw)
    qrm = qr(m)
    b = zeros(length(fes[1].refs))
    QRFixedEffectMatrix(fes, m, qrm, b)
end

function solve!(fep::QRFixedEffectMatrix, r::AbstractVector ; kwargs...) 
    # since \ needs a vector
    copyto!(fep.b, r)
    fep.qrm \ fep.b
end

##############################################################################
##
## Methods used by all matrix factorization
##
##############################################################################

# construct the sparse matrix of fixed effects A in  A'Ax = A'r
function sparse(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector)
    # construct model matrix A constituted by fixed effects
    nobs = length(fes[1].refs)
    N = length(fes) * nobs
    I = zeros(Int, N)
    J = similar(I)
    V = zeros(Float64, N)
    start = 0
    idx = 0
    for fe in fes
       for i in 1:length(fe.refs)
           idx += 1
           I[idx] = i
           J[idx] = start + fe.refs[i]
           V[idx] = fe.interaction[i] * sqrtw[i]
       end
       start += length(unique(fe.refs))
    end
    sparse(I, J, V)
end

# updates r as the residual of the projection of r on A
function solve_residuals!(r::AbstractVector, fep::Union{CholeskyFixedEffectMatrix, QRFixedEffectMatrix}; kwargs...)
    x = solve!(fep, r; kwargs...)
    mul!(r, fep.m, x, -1.0, 1.0)
    return r, 1, true
end

function solve_residuals!(X::AbstractMatrix, fep::Union{CholeskyFixedEffectMatrix, QRFixedEffectMatrix}; kwargs...)
    iterations = Vector{Int}(undef, size(X, 2))
    convergeds = Vector{Bool}(undef, size(X, 2))
    for j in 1:size(X, 2)
        #view disables simd
        X[:, j], iteration, converged = solve_residuals!(X[:, j], fep; kwargs...)
        iterations[j] = iteration
        convergeds[j] = converged
    end
    return X, iterations, convergeds
end


# solves A'Ax = A'r
# transform x from Vector (stacked vector of coefficients) 
# to Vector{Vector} (vector of coefficients for each categorical variable)
function _solve_coefficients!(r::AbstractVector, fep::Union{CholeskyFixedEffectMatrix, QRFixedEffectMatrix}; kwargs...)
    x = solve!(fep, r; kwargs...)
    out = Vector{eltype(r)}[]
    iend = 0
    for fe in fep.fes
        istart = iend + 1
        iend = istart + length(unique(fe.refs)) - 1
        push!(out, x[istart:iend])
    end
    return out, 1, true
end

get_fes(fep::Union{CholeskyFixedEffectMatrix, QRFixedEffectMatrix}) = fep.fes

