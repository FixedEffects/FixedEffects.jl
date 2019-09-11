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
## FixedEffectLSMR
##
## A is the model matrix of categorical variables
## normalized by diag(1/a1, ..., 1/aN) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## mul!
##
##############################################################################

struct FixedEffectLSMR <: AbstractFixedEffectMatrix
    fes::Vector{<:FixedEffect}
    scales::Vector{Vector{Float64}}
    caches::Vector{Vector{Float64}}
    x::FixedEffectVector{Float64}
    v::FixedEffectVector{Float64}
    h::FixedEffectVector{Float64}
    hbar::FixedEffectVector{Float64}
    u::Vector{Float64}
    sqrtw::AbstractVector{Float64}
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr}})
    scales = [_scale(fe, sqrtw) for fe in fes] 
    caches = [_cache(fe, scale, sqrtw) for (fe, scale) in zip(fes, scales)]
    x = FixedEffectVector(fes)
    v = FixedEffectVector(fes)
    h = FixedEffectVector(fes)
    hbar = FixedEffectVector(fes)
    u = zeros(length(fes[1].refs))
    return FixedEffectLSMR(fes, scales, caches, x, v, h, hbar, u, sqrtw)
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

eltype(fem::FixedEffectLSMR) = Float64
adjoint(fem::FixedEffectLSMR) = Adjoint(fem)

function size(fem::FixedEffectLSMR, dim::Integer)
    (dim == 1) ? length(fem.fes[1].refs) : (dim == 2) ? sum(fe.n for fe in fem.fes) : 1
end

function mul!(y::AbstractVector, fem::FixedEffectLSMR, 
                fev::FixedEffectVector, α::Number, β::Number)
    rmul!(y, β)
    for (x, fe, cache) in zip(fev.fes, fem.fes, fem.caches)
        demean!(y, x, fe.refs, α, cache)
    end
    return y
end

function demean!(y::AbstractVector, x::AbstractVector, refs::AbstractVector, 
                α::Number, cache::AbstractVector)
    @simd ivdep for i in eachindex(y)
        @inbounds y[i] += x[refs[i]] * α * cache[i]
    end
end

function mul!(fev::FixedEffectVector, Cfem::Adjoint{T, FixedEffectLSMR},
                y::AbstractVector, α::Number, β::Number) where {T}
    fem = adjoint(Cfem)
    rmul!(fev, β)
    for (x, fe, cache) in zip(fev.fes, fem.fes, fem.caches)
        mean!(x, fe.refs, y, α, cache)
    end
    return fev
end

function mean!(x::AbstractVector, refs::AbstractVector, y::AbstractVector, 
        α::Number, cache::AbstractVector)
   @simd ivdep for i in eachindex(y)
        @inbounds x[refs[i]] += y[i] * α * cache[i]
    end
end

function solve!(feM::FixedEffectLSMR, r::AbstractVector; 
    tol::Real = 1e-8, maxiter::Integer = 100_000)
    fill!(feM.x, 0.0)
    copyto!(feM.u, r)
    x, ch = lsmr!(feM.x, feM, feM.u, feM.v, feM.h, feM.hbar; 
        atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
    return div(ch.mvps, 2), ch.isconverged
end

##############################################################################
##
## Implement AbstractFixedEffectMatrix interface
##
##############################################################################\

function solve_residuals!(r::AbstractVector{Float64}, feM::FixedEffectLSMR; kwargs...)
    r .= r .* feM.sqrtw
    iterations, converged = solve!(feM, r; kwargs...)
    mul!(r, feM, feM.x, -1.0, 1.0)
    r .= r ./ feM.sqrtw
    return r, iterations, converged
end

function solve_coefficients!(r::AbstractVector{Float64}, feM::FixedEffectLSMR; kwargs...)
    r .= r .* feM.sqrtw
    iterations, converged = solve!(feM, r; kwargs...)
    for (fe, scale) in zip(feM.x.fes, feM.scales)
        fe .*=  scale
    end 
    normalize!(feM.x.fes, r, feM.fes; kwargs...), iterations, converged
end

##############################################################################
##
## LSMR Parallel
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct FixedEffectLSMRParallel{W} <: AbstractFixedEffectMatrix
    fes::Vector{<:FixedEffect}
    sqrtw::W
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, 
                ::Type{Val{:lsmr_parallel}})
    FixedEffectLSMRParallel(fes, sqrtw)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRParallel; kwargs...)
    solve_residuals!(r, FixedEffectMatrix(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectLSMRParallel; kwargs...)
    iterations = zeros(Int, size(X, 2))
    convergeds = zeros(Bool, size(X, 2))
    result = pmap(x -> solve_residuals!(x, feM;kwargs...), [X[:, j] for j in 1:size(X, 2)])
    for j in 1:size(X, 2)
        X[:, j] = result[j][1]
        iterations[j] =  result[j][2]
        convergeds[j] = result[j][3]
    end
    return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRParallel, ; kwargs...)
    solve_coefficients!(r, FixedEffectMatrix(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

##############################################################################
##
## LSMR MultiThreaded
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct FixedEffectLSMRThreads{W} <: AbstractFixedEffectMatrix
    fes::Vector{<:FixedEffect}
    sqrtw::W
end

FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_threads}}) = FixedEffectLSMRThreads(fes, sqrtw)

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRThreads; kwargs...)
    solve_residuals!(r, FixedEffectMatrix(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectLSMRThreads; kwargs...)
   iterations = zeros(Int, size(X, 2))
   convergeds = zeros(Bool, size(X, 2))
   Threads.@threads for j in 1:size(X, 2)
        X[:, j], iteration, converged = solve_residuals!(X[:, j], feM; kwargs...)
        iterations[j] = iteration
        convergeds[j] = converged
   end
   return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRThreads; kwargs...)
    solve_coefficients!(r, FixedEffectMatrix(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end
