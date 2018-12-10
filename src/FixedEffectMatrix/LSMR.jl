##############################################################################
## 
## Least Squares using LSMR
##
##############################################################################



##############################################################################
## 
## FixedEffectVector : vector x in A'Ax = A'b
##
## We define these methods used in lsmr! (duck typing): 
## copyto!, fill!, rmul!, axpy!, norm
##
##############################################################################

struct FixedEffectVector
    _::Vector{Vector{Float64}}
end

function FixedEffectVector(fes::Vector{<:FixedEffect})
    out = Vector{Float64}[]
    for fe in fes
        push!(out, zeros(fe.n))
    end
    return FixedEffectVector(out)
end

eltype(fem::FixedEffectVector) = Float64

length(fev::FixedEffectVector) = sum(length(x) for x in fev._)

function norm(fev::FixedEffectVector)
    sqrt(sum(sum(abs2, fe) for fe in  fev._))
end

function fill!(fev::FixedEffectVector, x)
    for fe in fev._
        fill!(fe, x)
    end
end

function rmul!(fev::FixedEffectVector, α::Number)
    for fe in fev._
        rmul!(fe, α)
    end
    return fev
end

function copyto!(fev2::FixedEffectVector, fev1::FixedEffectVector)
    for i in 1:length(fev1._)
        copyto!(fev2._[i], fev1._[i])
    end
    return fev2
end


function axpy!(α::Number, fev1::FixedEffectVector, fev2::FixedEffectVector)
    for i in 1:length(fev1._)
        axpy!(α, fev1._[i], fev2._[i])
    end
    return fev2
end



##############################################################################
## 
## _LSMRFixedEffectMatrix
##
## A is the model matrix of categorical variables
## normalized by diag(1/a1, ..., 1/aN) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## mul!
##
##############################################################################

struct _LSMRFixedEffectMatrix
    _::Vector{<:FixedEffect}
    m::Int
    n::Int
    scale::Vector{Vector{Float64}}
    cache::Vector{Vector{Float64}}
end

function _LSMRFixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector)
    m = length(fes[1].refs)
    n = sum(fe.n for fe in fes)
    scales = Vector{Float64}[]
    for i in 1:length(fes)
        push!(scales, cache(fes[i], sqrtw))
    end
    caches = Vector{Float64}[]
    for i in 1:length(fes)
        push!(caches, cache(fes[i], scales[i], sqrtw))
    end
    return _LSMRFixedEffectMatrix(fes, m, n, scales, caches)
end

function cache(x::FixedEffect, sqrtw)
    scale = zeros(x.n)
    for i in 1:length(x.refs)
        scale[x.refs[i]] += abs2(x.interaction[i] * sqrtw[i])
    end
    for i in 1:length(scale)
        scale[i] = scale[i] > 0.0 ? (1.0 / sqrt(scale[i])) : 0.0
    end
    return scale
end

function cache(fe::FixedEffect, scale, sqrtw::AbstractVector)
    out = zeros(Float64, length(fe.refs))
    @inbounds @simd ivdep for i in 1:length(out)
        out[i] = scale[fe.refs[i]] * fe.interaction[i] * sqrtw[i]
    end
    return out
end

eltype(fem::_LSMRFixedEffectMatrix) = Float64

size(fem::_LSMRFixedEffectMatrix, dim::Integer) = (dim == 1) ? fem.m :
                                            (dim == 2) ? fem.n : 1
Base.adjoint(fem) = Adjoint(fem)

function mul!(y::AbstractVector{Float64}, fem::_LSMRFixedEffectMatrix, fev::FixedEffectVector, α::Number, β::Number)
    safe_rmul!(y, β)
    for i in 1:length(fev._)
        helperN!(α, fem._[i], fev._[i], y, fem.cache[i])
    end
    return y
end
# Define x -> A * x
function helperN!(α::Number, fe::FixedEffect, 
    x::Vector{Float64}, y::AbstractVector{Float64}, cache::Vector{Float64})
    @inbounds @simd ivdep for i in 1:length(y)
        y[i] += α * x[fe.refs[i]] * cache[i]
    end
end

function mul!(fev::FixedEffectVector, Cfem::Adjoint{T, _LSMRFixedEffectMatrix}, y::AbstractVector{Float64}, α::Number, β::Number) where {T}
    fem = adjoint(Cfem)
    safe_rmul!(fev, β)
    for i in 1:length(fev._)
        helperC!(α, fem._[i], y, fev._[i], fem.cache[i])
    end
    return fev
end

# Define x -> A' * x
function helperC!(α::Number, fe::FixedEffect, 
                        y::AbstractVector{Float64}, x::Vector{Float64}, cache::Vector{Float64})
    @inbounds @simd ivdep for i in 1:length(y)
        x[fe.refs[i]] += α * y[i] * cache[i]
    end
end

function safe_rmul!(x, β)
    if !(β ≈ 1.0)
        β ≈ 0.0 ? fill!(x, zero(eltype(x))) : rmul!(x, β)
    end
end

##############################################################################
##
## LSMRFixedEffectMatrix is a wrapper around a _LSMRFixedEffectMatrix 
## with some storage arrays used when solving (A'A)X = A'y 
##
##############################################################################

struct LSMRFixedEffectMatrix <: FixedEffectMatrix
    m::_LSMRFixedEffectMatrix
    x::FixedEffectVector
    v::FixedEffectVector
    h::FixedEffectVector
    hbar::FixedEffectVector
    u::Vector{Float64}
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr}})
    m = _LSMRFixedEffectMatrix(fes, sqrtw)
    x = FixedEffectVector(fes)
    v = FixedEffectVector(fes)
    h = FixedEffectVector(fes)
    hbar = FixedEffectVector(fes)
    u = Array{Float64}(undef, size(m, 1))
    return LSMRFixedEffectMatrix(m, x, v, h, hbar, u)
end

get_fes(fep::LSMRFixedEffectMatrix) = fep.m._

function solve!(fep::LSMRFixedEffectMatrix, r::AbstractVector{Float64}; 
    tol::Real = 1e-8, maxiter::Integer = 100_000)
    fill!(fep.x, zero(Float64))
    copyto!(fep.u, r)
    x, ch = lsmr!(fep.x, fep.m, fep.u, fep.v, fep.h, fep.hbar; 
        atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
    return div(ch.mvps, 2), ch.isconverged
end

function solve_residuals!(r::AbstractVector{Float64}, fep::LSMRFixedEffectMatrix; kwargs...)
    iterations, converged = solve!(fep, r; kwargs...)
    mul!(r, fep.m, fep.x, -1.0, 1.0)
    return r, iterations, converged
end

function _solve_coefficients!(r::AbstractVector{Float64}, fep::LSMRFixedEffectMatrix; kwargs...)
    iterations, converged = solve!(fep, r; kwargs...)
    for i in 1:length(fep.x._)
        fep.x._[i] .= fep.x._[i] .* fep.m.scale[i]
    end
    return fep.x._, iterations, converged
end


##############################################################################
##
## LSMR Parallel
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct LSMRParallelFixedEffectMatrix <: FixedEffectMatrix
    fes::Vector{<:FixedEffect}
    sqrtw::AbstractVector
end

FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_parallel}}) = LSMRParallelFixedEffectMatrix(fes, sqrtw)
get_fes(fep::LSMRParallelFixedEffectMatrix) = fep.fes


function solve_residuals!(X::AbstractMatrix{Float64}, fep::LSMRParallelFixedEffectMatrix; kwargs...)
    iterations = Vector{Int}(undef, size(X, 2))
    convergeds = Vector{Bool}(undef, size(X, 2))
    result = pmap(x -> solve_residuals!(x, fep;kwargs...), [X[:, j] for j in 1:size(X, 2)])
    for j in 1:size(X, 2)
        X[:, j] = result[j][1]
        iterations[j] =  result[j][2]
        convergeds[j] = result[j][3]
    end
    return X, iterations, convergeds
end

function solve_residuals!(r::AbstractVector{Float64}, fep::LSMRParallelFixedEffectMatrix; kwargs...)
    solve_residuals!(r, FixedEffectMatrix(fep.fes, fep.sqrtw, Val{:lsmr}); kwargs...)
end

function _solve_coefficients!(r::AbstractVector{Float64}, fep::LSMRParallelFixedEffectMatrix, ; kwargs...)
    _solve_coefficients!(r, FixedEffectMatrix(fep.fes, fep.sqrtw, Val{:lsmr}); kwargs...)
end


##############################################################################
##
## LSMR MultiThreaded
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct LSMRThreadslFixedEffectMatrix <: FixedEffectMatrix
    fes::Vector{<:FixedEffect}
    sqrtw::AbstractVector
end

FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_threads}}) = LSMRThreadslFixedEffectMatrix(fes, sqrtw)
get_fes(fep::LSMRThreadslFixedEffectMatrix) = fep.fes


function solve_residuals!(X::AbstractMatrix{Float64}, fep::LSMRThreadslFixedEffectMatrix; kwargs...)
   iterations = Vector{Int}(undef, size(X, 2))
   convergeds = Vector{Bool}(undef, size(X, 2))
   Threads.@threads for j in 1:size(X, 2)
        r, iteration, converged = solve_residuals!(view(X, :, j), fep; kwargs...)
        iterations[j] = iteration
        convergeds[j] = converged
   end
   return X, iterations, convergeds
end

function solve_residuals!( r::AbstractVector{Float64}, fep::LSMRThreadslFixedEffectMatrix; kwargs...)
    solve_residuals!(r, FixedEffectMatrix(fep.fes, fep.sqrtw, Val{:lsmr}); kwargs...)
end
function _solve_coefficients!(r::AbstractVector{Float64}, fep::LSMRThreadslFixedEffectMatrix; kwargs...)
    _solve_coefficients!(r, FixedEffectMatrix(fep.fes, fep.sqrtw, Val{:lsmr}); kwargs...)
end

