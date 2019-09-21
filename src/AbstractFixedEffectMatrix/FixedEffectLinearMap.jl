##############################################################################
## 
## FixedEffectCoefficients : vector x in A'Ax = A'b
##
## We define these methods used in lsmr! (duck typing): 
## copyto!, fill!, rmul!, axpy!, norm
##
##############################################################################

struct FixedEffectCoefficients{T}
    x::Vector{<:AbstractVector{T}}
end
Base.iterate(xs::FixedEffectCoefficients) = iterate(xs.x)
Base.iterate(xs::FixedEffectCoefficients, state) = iterate(xs.x, state)


eltype(xs::FixedEffectCoefficients{T}) where {T} = T
length(xs::FixedEffectCoefficients) = sum(length(x) for x in xs)
norm(xs::FixedEffectCoefficients) = sqrt(sum(sum(abs2, x) for x in xs))

function fill!(xs::FixedEffectCoefficients, α::Number)
    for x in xs
        fill!(x, α)
    end
    return xs
end

function rmul!(xs::FixedEffectCoefficients, α::Number)
    for x in xs
        rmul!(x, α)
    end
    return xs
end

function copyto!(xs1::FixedEffectCoefficients, xs2::FixedEffectCoefficients)
    for (x1, x2) in zip(xs1, xs2)
        copyto!(x1, x2)
    end
    return xs1
end

function axpy!(α::Number, xs1::FixedEffectCoefficients, xs2::FixedEffectCoefficients)
    for (x1, x2) in zip(xs1, xs2)
        axpy!(α, x1, x2)
    end
    return xs2
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

struct FixedEffectLSMR{T} <: AbstractFixedEffectMatrix{T}
    fes::Vector{<:FixedEffect}
    scales::FixedEffectCoefficients{T}
    caches::Vector{<:AbstractVector}
    xs::FixedEffectCoefficients{T}
    v::FixedEffectCoefficients{T}
    h::FixedEffectCoefficients{T}
    hbar::FixedEffectCoefficients{T}
    u::AbstractVector{T}
    sqrtw::AbstractVector{T}
end

adjoint(fem::FixedEffectLSMR) = Adjoint(fem)

function size(fem::FixedEffectLSMR, dim::Integer)
    (dim == 1) ? length(fem.fes[1].refs) : (dim == 2) ? sum(fe.n for fe in fem.fes) : 1
end

function mul!(y::AbstractVector, fem::FixedEffectLSMR, 
                fecoefs::FixedEffectCoefficients, α::Number, β::Number)
    rmul!(y, β)
    for (fecoef, fe, cache) in zip(fecoefs, fem.fes, fem.caches)
        demean!(y, fecoef, fe.refs, α, cache)
    end
    return y
end

function demean!(y::AbstractVector, fecoef::AbstractVector, refs::AbstractVector, 
                α::Number, cache::AbstractVector)
    @simd ivdep for i in eachindex(y)
        @inbounds y[i] += fecoef[refs[i]] * α * cache[i]
    end
end

function mul!(fecoefs::FixedEffectCoefficients, Cfem::Adjoint{T, FixedEffectLSMR{T}},
                y::AbstractVector, α::Number, β::Number) where {T}
    fem = adjoint(Cfem)
    rmul!(fecoefs, β)
    for (fecoef, fe, cache) in zip(fecoefs, fem.fes, fem.caches)
        mean!(fecoef, fe.refs, y, α, cache)
    end
    return fecoefs
end

function mean!(fecoef::AbstractVector, refs::AbstractVector, y::AbstractVector, 
        α::Number, cache::AbstractVector)
   @simd ivdep for i in eachindex(y)
        @inbounds fecoef[refs[i]] += y[i] * α * cache[i]
    end
end

function solve!(feM::FixedEffectLSMR, r::AbstractVector; 
    tol::Real = 1e-8, maxiter::Integer = 100_000)
    fill!(feM.xs, 0.0)
    copyto!(feM.u, r)
    x, ch = lsmr!(feM.xs, feM, feM.u, feM.v, feM.h, feM.hbar; 
        atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
    return div(ch.mvps, 2), ch.isconverged
end

##############################################################################
##
## Implement AbstractFixedEffectMatrix interface
##
##############################################################################\

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr}})
    n = length(sqrtw)
    scales = FixedEffectCoefficients([scale!(zeros(fe.n), fe.refs, fe.interaction, sqrtw) for fe in fes])
    caches = [cache!(zeros(n), fe.refs, fe.interaction, scale, sqrtw) for (fe, scale) in zip(fes, scales)]
    xs = FixedEffectCoefficients([zeros(fe.n) for fe in fes])
    v = FixedEffectCoefficients([zeros(fe.n) for fe in fes])
    h = FixedEffectCoefficients([zeros(fe.n) for fe in fes])
    hbar = FixedEffectCoefficients([zeros(fe.n) for fe in fes])
    u = zeros(n)
    return FixedEffectLSMR(fes, scales, caches, xs, v, h, hbar, u, sqrtw)
end

function scale!(fecoef::AbstractVector, refs::AbstractVector, interaction::AbstractVector, sqrtw::AbstractVector)
    @inbounds @simd for i in eachindex(refs)
        fecoef[refs[i]] += abs2(interaction[i] * sqrtw[i])
    end
    fecoef .= 1.0 ./ sqrt.(fecoef)
end

function cache!(y::AbstractVector, refs::AbstractVector, interaction::AbstractVector, fecoef::AbstractVector, sqrtw::AbstractVector)
    @inbounds @simd for i in eachindex(y)
        y[i] = fecoef[refs[i]] * interaction[i] * sqrtw[i]
    end
    return y
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMR; kwargs...)
    r .*= feM.sqrtw
    iterations, converged = solve!(feM, r; kwargs...)
    mul!(r, feM, feM.xs, -1.0, 1.0)
    r ./=  feM.sqrtw
    return r, iterations, converged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMR; kwargs...)
	iterations, converged = _solve_coefficients!(r, feM)
    full(normalize!(feM.xs.x, feM.fes; kwargs...), feM.fes), iterations, converged
end

function _solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMR; kwargs...)
	r .*= feM.sqrtw
	iterations, converged = solve!(feM, r; kwargs...)
	for (x, scale) in zip(feM.xs, feM.scales)
	    x .*=  scale
	end
	iterations, converged
end