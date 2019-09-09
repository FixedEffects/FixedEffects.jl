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

struct FixedEffectVector{T}
    fes::Vector{Vector{T}}
end

function FixedEffectVector(fes::Vector{<:FixedEffect})
    FixedEffectVector([zeros(fe.n) for fe in fes])
end

eltype(fem::FixedEffectVector{T}) where {T} = T
length(fev::FixedEffectVector) = sum(length(x) for x in fev.fes)
norm(fev::FixedEffectVector) = sqrt(sum(sum(abs2, fe) for fe in  fev.fes))

function fill!(fev::FixedEffectVector, x)
    for fe in fev.fes
        fill!(fe, x)
    end
    return fev
end

function rmul!(fev::FixedEffectVector, α::Number)
    for fe in fev.fes
        rmul!(fe, α)
    end
    return fev
end

function copyto!(fev1::FixedEffectVector, fev2::FixedEffectVector)
    for (fe1, fe2) in zip(fev1.fes, fev2.fes)
        copyto!(fe1, fe2)
    end
    return fev1
end

function axpy!(α::Number, fev1::FixedEffectVector, fev2::FixedEffectVector)
    for (fe1, fe2) in zip(fev1.fes, fev2.fes)
        axpy!(α, fe1, fe2)
    end
    return fev2
end

##############################################################################
## 
## PreconditionnedLSMRFixedEffectMatrix
##
## A is the model matrix of categorical variables
## normalized by diag(1/a1, ..., 1/aN) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## mul!
##
##############################################################################

struct PreconditionnedLSMRFixedEffectMatrix
    fes::Vector{<:FixedEffect}
    scales::Vector{Vector{Float64}}
    caches::Vector{Vector{Float64}}
end

function PreconditionnedLSMRFixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector)
    scales = [_scale(fe, sqrtw) for fe in fes] 
    caches = [_cache(fe, scale, sqrtw) for (fe, scale) in zip(fes, scales)]
    return PreconditionnedLSMRFixedEffectMatrix(fes, scales, caches)
end

function _scale(x::FixedEffect, sqrtw)
    out = zeros(x.n)
    for i in eachindex(x.refs)
        out[x.refs[i]] += abs2(x.interaction[i] * sqrtw[i])
    end
    for i in eachindex(out)
        out[i] = out[i] > 0.0 ? (1.0 / sqrt(out[i])) : 0.0
    end
    return out
end

function _cache(fe::FixedEffect, scale, sqrtw::AbstractVector)
    out = zeros(length(fe.refs))
    @inbounds @simd for i in eachindex(out)
        out[i] = scale[fe.refs[i]] * fe.interaction[i] * sqrtw[i]
    end
    return out
end

eltype(fem::PreconditionnedLSMRFixedEffectMatrix) = Float64
adjoint(fem::PreconditionnedLSMRFixedEffectMatrix) = Adjoint(fem)

function size(fem::PreconditionnedLSMRFixedEffectMatrix, dim::Integer)
    (dim == 1) ? length(fem.fes[1].refs) : (dim == 2) ? sum(fe.n for fe in fem.fes) : 1
end


function mul!(y::AbstractVector, fem::PreconditionnedLSMRFixedEffectMatrix, 
                fev::FixedEffectVector, α::Number, β::Number)
    rmul!(y, β)
    for (x, fe, cache) in zip(fev.fes, fem.fes, fem.caches)
        helperN!(y, x, fe, α, cache)
    end
    return y
end

function helperN!(y::AbstractVector, x::AbstractVector, fe::FixedEffect, 
                α::Number, cache::AbstractVector)
    @inbounds @simd ivdep for i in eachindex(y)
        y[i] += x[fe.refs[i]] * α * cache[i]
    end
end


function mul!(fev::FixedEffectVector, Cfem::Adjoint{T, PreconditionnedLSMRFixedEffectMatrix},
                y::AbstractVector, α::Number, β::Number) where {T}
    fem = adjoint(Cfem)
    rmul!(fev, β)
    for (x, fe, cache) in zip(fev.fes, fem.fes, fem.caches)
        helperC!(x, fe, y, α, cache)
    end
    return fev
end

function helperC!(x::AbstractVector, fe::FixedEffect, y::AbstractVector, 
        α::Number, cache::AbstractVector)
    @inbounds @simd ivdep for i in eachindex(y)
        x[fe.refs[i]] += y[i] * α * cache[i]
    end
end

##############################################################################
##
## LSMRFixedEffectMatrix is a wrapper around a PreconditionnedLSMRFixedEffectMatrix 
## with some storage arrays used when solving (A'A)X = A'y 
##
##############################################################################

struct LSMRFixedEffectMatrix <: AbstractFixedEffectMatrix
    m::PreconditionnedLSMRFixedEffectMatrix
    x::FixedEffectVector{Float64}
    v::FixedEffectVector{Float64}
    h::FixedEffectVector{Float64}
    hbar::FixedEffectVector{Float64}
    u::Vector{Float64}
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr}})
    m = PreconditionnedLSMRFixedEffectMatrix(fes, sqrtw)
    x = FixedEffectVector(fes)
    v = FixedEffectVector(fes)
    h = FixedEffectVector(fes)
    hbar = FixedEffectVector(fes)
    u = zeros(size(m, 1))
    return LSMRFixedEffectMatrix(m, x, v, h, hbar, u)
end

get_fes(fep::LSMRFixedEffectMatrix) = fep.m.fes

function solve!(fep::LSMRFixedEffectMatrix, r::AbstractVector; 
    tol::Real = 1e-8, maxiter::Integer = 100_000)
    fill!(fep.x, 0.0)
    copyto!(fep.u, r)
    x, ch = lsmr!(fep.x, fep.m, fep.u, fep.v, fep.h, fep.hbar; 
        atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
    return div(ch.mvps, 2), ch.isconverged
end

function solve_residuals!(r::AbstractVector, fep::LSMRFixedEffectMatrix; kwargs...)
    iterations, converged = solve!(fep, r; kwargs...)
    mul!(r, fep.m, fep.x, -1.0, 1.0)
    return r, iterations, converged
end
function solve_residuals!(X::AbstractMatrix, fep::LSMRFixedEffectMatrix; kwargs...)
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



function _solve_coefficients!(r::AbstractVector, fep::LSMRFixedEffectMatrix; kwargs...)
    iterations, converged = solve!(fep, r; kwargs...)
    for (fe, scale) in zip(fep.x.fes, fep.m.scales)
        fe .*=  scale
    end
    return fep.x.fes, iterations, converged
end


##############################################################################
##
## LSMR Parallel
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct LSMRParallelFixedEffectMatrix{W} <: AbstractFixedEffectMatrix
    fes::Vector{<:FixedEffect}
    sqrtw::W
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, 
                ::Type{Val{:lsmr_parallel}})
    LSMRParallelFixedEffectMatrix(fes, sqrtw)
end
get_fes(fep::LSMRParallelFixedEffectMatrix) = fep.fes


function solve_residuals!(X::AbstractMatrix, fep::LSMRParallelFixedEffectMatrix; kwargs...)
    iterations = zeros(Int, size(X, 2))
    convergeds = zeros(Bool, size(X, 2))
    result = pmap(x -> solve_residuals!(x, fep;kwargs...), [X[:, j] for j in 1:size(X, 2)])
    for j in 1:size(X, 2)
        X[:, j] = result[j][1]
        iterations[j] =  result[j][2]
        convergeds[j] = result[j][3]
    end
    return X, iterations, convergeds
end

function solve_residuals!(r::AbstractVector, fep::LSMRParallelFixedEffectMatrix; kwargs...)
    solve_residuals!(r, FixedEffectMatrix(fep.fes, fep.sqrtw, Val{:lsmr}); kwargs...)
end

function _solve_coefficients!(r::AbstractVector, fep::LSMRParallelFixedEffectMatrix, ; kwargs...)
    _solve_coefficients!(r, FixedEffectMatrix(fep.fes, fep.sqrtw, Val{:lsmr}); kwargs...)
end


##############################################################################
##
## LSMR MultiThreaded
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct LSMRThreadslFixedEffectMatrix{W} <: AbstractFixedEffectMatrix
    fes::Vector{<:FixedEffect}
    sqrtw::W
end

FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_threads}}) = LSMRThreadslFixedEffectMatrix(fes, sqrtw)
get_fes(fep::LSMRThreadslFixedEffectMatrix) = fep.fes

function solve_residuals!(X::AbstractMatrix, fep::LSMRThreadslFixedEffectMatrix; kwargs...)
   iterations = zeros(Int, size(X, 2))
   convergeds = zeros(Bool, size(X, 2))
   Threads.@threads for j in 1:size(X, 2)
        X[:, j], iteration, converged = solve_residuals!(X[:, j], fep; kwargs...)
        iterations[j] = iteration
        convergeds[j] = converged
   end
   return X, iterations, convergeds
end

function solve_residuals!( r::AbstractVector, fep::LSMRThreadslFixedEffectMatrix; kwargs...)
    solve_residuals!(r, FixedEffectMatrix(fep.fes, fep.sqrtw, Val{:lsmr}); kwargs...)
end
function _solve_coefficients!(r::AbstractVector, fep::LSMRThreadslFixedEffectMatrix; kwargs...)
    _solve_coefficients!(r, FixedEffectMatrix(fep.fes, fep.sqrtw, Val{:lsmr}); kwargs...)
end

