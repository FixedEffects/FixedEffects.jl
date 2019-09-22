##############################################################################
##
## Use by all matrix factorization
##
##############################################################################

# construct the sparse matrix of fixed effects A in  A'Ax = A'r
function AbstractFixedEffectMatrix{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector,  ::Type{Val{:CSC}}) where {T}
    # construct model matrix A constituted by fixed effects
    nobs = length(fes[1].refs)
    N = length(fes) * nobs
    I = zeros(Int, N)
    J = similar(I)
    V = zeros(T, N)
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

struct FixedEffectCholesky{CholT, T, N, Tw} <: AbstractFixedEffectMatrix{T}
    fes::Vector{<:FixedEffect}
    m::SparseMatrixCSC{T, N}
    cholm::CholT
    x::Vector{T}
    r::Vector{T}
    sqrtw::Tw
end

function AbstractFixedEffectMatrix{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:cholesky}}) where {T}
    m = AbstractFixedEffectMatrix{T}(fes, sqrtw, Val{:CSC})
    cholm = cholesky(Symmetric(m' * m))
    total_len = sum(length(unique(fe.refs)) for fe in fes)
    FixedEffectCholesky(fes, m, cholm, zeros(T, total_len), zeros(T, length(sqrtw)), sqrtw)
end

function solve!(feM::FixedEffectCholesky, r::AbstractVector; kwargs...)
    feM.cholm \ mul!(feM.x, feM.m', r)
end

##############################################################################
##
## Least Squares via QR Factorization
##
##############################################################################

struct FixedEffectQR{QRT, T, N, Tw} <: AbstractFixedEffectMatrix{T}
    fes::Vector{<:FixedEffect}
    m::SparseMatrixCSC{T, N}
    qrm::QRT
    r::Vector{T}
    sqrtw::Tw
end

function AbstractFixedEffectMatrix{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:qr}}) where {T}
    m = AbstractFixedEffectMatrix{T}(fes, sqrtw, Val{:CSC})
    qrm = qr(m)
    r = zeros(T, length(sqrtw))
    FixedEffectQR(fes, m, qrm, r, sqrtw)
end

function solve!(feM::FixedEffectQR, r::AbstractVector ; kwargs...) 
    feM.qrm \ r
end


##############################################################################
##
## Implement AbstractFixedEffectMatrix interface
##
##############################################################################

# updates r as the residual of the projection of r on A
function solve_residuals!(r::AbstractVector, feM::Union{FixedEffectCholesky, FixedEffectQR}; kwargs...)
    copyto!(feM.r, r)
    feM.r .*= feM.sqrtw
    x = solve!(feM, feM.r; kwargs...)
    mul!(feM.r, feM.m, x, -1.0, 1.0)
    feM.r ./= feM.sqrtw
    copyto!(r, feM.r)
    return r, 1, true
end

# solves A'Ax = A'r
# transform x from Vector (stacked vector of coefficients) 
# to Vector{Vector} (vector of coefficients for each categorical variable)
function solve_coefficients!(r::AbstractVector, feM::Union{FixedEffectCholesky, FixedEffectQR}; kwargs...)
    copyto!(feM.r, r)
    feM.r .*= feM.sqrtw
    x = solve!(feM, feM.r; kwargs...)
    out = Vector{eltype(r)}[]
    iend = 0
    for fe in feM.fes
        istart = iend + 1
        iend = istart + length(Set(fe.refs)) - 1
        push!(out, x[istart:iend])
    end
    full(normalize!(out, feM.fes; kwargs...), feM.fes), 1, true
end
