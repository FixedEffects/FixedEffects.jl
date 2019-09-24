struct FixedEffectSolverLSMRParallel{T} <: AbstractFixedEffectSolver{T}
    fes::Vector{<:FixedEffect}
    sqrtw::AbstractVector{T}
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, 
                ::Type{Val{:lsmr_parallel}}) where {T}
    FixedEffectSolverLSMRParallel{T}(fes, sqrtw)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectSolverLSMRParallel{T}; kwargs...) where {T}
    solve_residuals!(r, AbstractFixedEffectSolver{T}(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectSolverLSMRParallel; kwargs...)
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

function solve_coefficients!(r::AbstractVector, feM::FixedEffectSolverLSMRParallel{T}, ; kwargs...) where {T}
    solve_coefficients!(r, AbstractFixedEffectSolver{T}(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end