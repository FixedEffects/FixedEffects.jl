##############################################################################
##
## LSMR MultiThreaded
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
## This is because each thread need to have its own cache / u / w, etc
##
##############################################################################

struct FixedEffectSolverLSMRThreads{T} <: AbstractFixedEffectSolver{T}
    fes::Vector{<:FixedEffect}
    sqrtw::AbstractVector
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_threads}}) where {T}
    FixedEffectSolverLSMRThreads{T}(fes, sqrtw)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectSolverLSMRThreads{T}; kwargs...) where {T}
    solve_residuals!(r, AbstractFixedEffectSolver{T}(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectSolverLSMRThreads; kwargs...)
   iterations = zeros(Int, size(X, 2))
   convergeds = zeros(Bool, size(X, 2))
   Threads.@threads for j in 1:size(X, 2)
        _, iteration, converged = solve_residuals!(view(X, :, j), feM; kwargs...)
        iterations[j] = iteration
        convergeds[j] = converged
   end
   return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectSolverLSMRThreads{T}; kwargs...) where {T}
    solve_coefficients!(r, AbstractFixedEffectSolver{T}(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end