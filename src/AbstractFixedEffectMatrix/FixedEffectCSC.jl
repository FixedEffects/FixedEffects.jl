##############################################################################
##
## Use by all matrix factorization
##
##############################################################################

# construct the sparse matrix of fixed effects A in  A'Ax = A'r
function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector,  ::Type{Val{:CSC}})
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

##############################################################################
##
## Least Squares via Cholesky Factorization
##
##############################################################################

struct FixedEffectCholesky{T, V, N, Tw} <: AbstractFixedEffectMatrix{V}
    fes::Vector{<:FixedEffect}
    m::SparseMatrixCSC{V, N}
    cholm::T
    x::Vector{V}
    sqrtw::Tw
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:cholesky}})
    m = FixedEffectMatrix(fes, sqrtw, Val{:CSC})
    cholm = cholesky(Symmetric(m' * m))
    total_len = sum(length(unique(fe.refs)) for fe in fes)
    FixedEffectCholesky(fes, m, cholm, zeros(total_len), sqrtw)
end

function solve!(feM::FixedEffectCholesky, r::AbstractVector; kwargs...)
    feM.cholm \ mul!(feM.x, feM.m', r)
end

##############################################################################
##
## Least Squares via QR Factorization
##
##############################################################################

struct FixedEffectQR{T, V, N, Tw} <: AbstractFixedEffectMatrix{V}
    fes::Vector{<:FixedEffect}
    m::SparseMatrixCSC{V, N}
    qrm::T
    b::Vector{V}
    sqrtw::Tw
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:qr}})
    m = FixedEffectMatrix(fes, sqrtw, Val{:CSC})
    qrm = qr(m)
    b = zeros(length(fes[1].refs))
    FixedEffectQR(fes, m, qrm, b, sqrtw)
end

function solve!(feM::FixedEffectQR, r::AbstractVector ; kwargs...) 
    # since \ needs a vector
    copyto!(feM.b, r)
    feM.qrm \ feM.b
end


##############################################################################
##
## Implement AbstractFixedEffectMatrix interface
##
##############################################################################

# updates r as the residual of the projection of r on A
function solve_residuals!(r::AbstractVector, feM::Union{FixedEffectCholesky, FixedEffectQR}; kwargs...)
    r .= r .* feM.sqrtw
    x = solve!(feM, r; kwargs...)
    mul!(r, feM.m, x, -1.0, 1.0)
    r .= r./ feM.sqrtw
    return r, 1, true
end

# solves A'Ax = A'r
# transform x from Vector (stacked vector of coefficients) 
# to Vector{Vector} (vector of coefficients for each categorical variable)
function solve_coefficients!(r::AbstractVector, feM::Union{FixedEffectCholesky, FixedEffectQR}; kwargs...)
    r .= r .* feM.sqrtw
    x = solve!(feM, r; kwargs...)
    out = Vector{eltype(r)}[]
    iend = 0
    for fe in feM.fes
        istart = iend + 1
        iend = istart + length(unique(fe.refs)) - 1
        push!(out, x[istart:iend])
    end
    full(normalize!(out, feM.fes; kwargs...), feM.fes), 1, true
end
