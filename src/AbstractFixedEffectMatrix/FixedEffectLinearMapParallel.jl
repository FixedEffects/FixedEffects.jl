##############################################################################
##
## LSMR Parallel
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct FixedEffectLSMRParallel{T} <: AbstractFixedEffectMatrix{T}
    fes::Vector{<:FixedEffect}
    sqrtw::AbstractVector{T}
end

function AbstractFixedEffectMatrix{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, 
                ::Type{Val{:lsmr_parallel}}) where {T}
    FixedEffectLSMRParallel{T}(fes, sqrtw)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRParallel{T}; kwargs...) where {T}
    solve_residuals!(r, AbstractFixedEffectMatrix{T}(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectLSMRParallel; kwargs...)
    iterations = zeros(Int, size(X, 2))
    convergeds = zeros(Bool, size(X, 2))
    result = pmap(x -> solve_residuals!(x, feM; kwargs...), [X[:, j] for j in 1:size(X, 2)])
    for j in 1:size(X, 2)
        X[:, j] = result[j][1]
        iterations[j] =  result[j][2]
        convergeds[j] = result[j][3]
    end
    return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRParallel{T}, ; kwargs...) where {T}
    solve_coefficients!(r, AbstractFixedEffectMatrix{T}(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

##############################################################################
##
## LSMR MultiThreaded
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct FixedEffectLSMRThreads{T} <: AbstractFixedEffectMatrix{T}
    fes::Vector{<:FixedEffect}
    sqrtw::AbstractVector
end

function AbstractFixedEffectMatrix{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_threads}}) where {T}
    FixedEffectLSMRThreads{T}(fes, sqrtw)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRThreads{T}; kwargs...) where {T}
    solve_residuals!(r, AbstractFixedEffectMatrix{T}(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectLSMRThreads; kwargs...)
   iterations = zeros(Int, size(X, 2))
   convergeds = zeros(Bool, size(X, 2))
   Threads.@threads for j in 1:size(X, 2)
        _, iteration, converged = solve_residuals!(view(X, :, j), feM; kwargs...)
        iterations[j] = iteration
        convergeds[j] = converged
   end
   return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRThreads{T}; kwargs...) where {T}
    solve_coefficients!(r, AbstractFixedEffectMatrix{T}(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end