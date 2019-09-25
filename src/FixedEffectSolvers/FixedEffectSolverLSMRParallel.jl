##############################################################################
##
## LSMR MultiCores
##
##############################################################################

struct FixedEffectSolverLSMRCores{T} <: AbstractFixedEffectSolver{T}
    fes::Vector{<:FixedEffect}
    sqrtw::AbstractVector{T}
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, 
                ::Type{Val{:lsmr_cores}}) where {T}
    FixedEffectSolverLSMRCores{T}(fes, sqrtw)
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, 
                ::Type{Val{:lsmr_parallel}}) where {T}
    info(":lsmr_parallel is deprecated. Use :lsmr_cores")
    AbstractFixedEffectSolver{T}(fes, sqrtw, Val{:lsmr_cores})
end


function solve_residuals!(r::AbstractVector, feM::FixedEffectSolverLSMRCores{T}; kwargs...) where {T}
    solve_residuals!(r, AbstractFixedEffectSolver{T}(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectSolverLSMRCores; kwargs...)
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

function solve_coefficients!(r::AbstractVector, feM::FixedEffectSolverLSMRCores{T}, ; kwargs...) where {T}
    solve_coefficients!(r, AbstractFixedEffectSolver{T}(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

##############################################################################
##
## LSMR MultiThreads
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
## This is because each thread need to have its own cache / u / w, etc
##
##############################################################################

struct FixedEffectSolverLSMRThreads{T} <: AbstractFixedEffectSolver{T}
    x::Vector{FixedEffectSolverLSMR{T}}
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_threads}}) where {T}
    FixedEffectSolverLSMRThreads([AbstractFixedEffectSolver{T}(fes, sqrtw, Val{:lsmr}) for i in 1:Threads.nthreads()])
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectSolverLSMRThreads; kwargs...)
    solve_residuals!(r, feM.x[1]; kwargs...)
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectSolverLSMRThreads; kwargs...)
   iterations = zeros(Int, size(X, 2))
   convergeds = zeros(Bool, size(X, 2))
   Threads.@threads for j in 1:size(X, 2)
        _, iteration, converged = solve_residuals!(view(X, :, j), feM.x[Threads.threadid()]; kwargs...)
        iterations[j] = iteration
        convergeds[j] = converged
   end
   return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectSolverLSMRThreads; kwargs...)
    solve_coefficients!(r, feM.x[1]; kwargs...)
end