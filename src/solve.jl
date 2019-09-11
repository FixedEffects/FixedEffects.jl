
# this type must defined solve_residuals!, solve_coefficients!, and fixedeffects
abstract type AbstractFixedEffectMatrix end

"""

Solve a least square problem for a set of FixedEffects

`solve_residuals!(y, fes, weights; method = :lsmr, maxiter = 10000, tol = 1e-8)`

### Arguments
* `y` : A `AbstractVector`
* `fes`: A `Vector{<:FixedEffect}`
* `weights`: A `AbstractWeights`
* `method` : A `Symbol` for the method. Choices are :lsmr, :lsmr_threads, :lsmr_parallel, :qr and :cholesky
* `maxiter` : Maximum number of iterations
* `tol` : Tolerance


### Returns
* `res` :  Residual of the least square problem
* `iterations`: Number of iterations
* `converged`: Did the algorithm converge?

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
solve_residuals!(rand(10), [FixedEffect(p1), FixedEffect(p2)])
```
"""
function solve_residuals!(y::Union{AbstractVector, AbstractMatrix}, fes::Vector{<: FixedEffect}, weights::AbstractWeights = Weights(Ones{eltype(y)}(size(y, 1))); method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    any(ismissing.(fes)) && error("Some FixedEffect has a missing value for reference or interaction")
    feM = FixedEffectMatrix(fes, sqrt.(weights.values), Val{method})
    y, iteration, converged = solve_residuals!(y, feM; maxiter = maxiter, tol = tol)
end

function solve_residuals!(X::AbstractMatrix, feM::AbstractFixedEffectMatrix; kwargs...)
    iterations = Vector{Int}(undef, size(X, 2))
    convergeds = Vector{Bool}(undef, size(X, 2))
    for j in 1:size(X, 2)
        #view disables simd
        X[:, j], iteration, converged = solve_residuals!(X[:, j], feM; kwargs...)
        iterations[j] = iteration
        convergeds[j] = converged
    end
    return X, iterations, convergeds
end
##############################################################################
##
## solve_coefficients!
##
## Fixed effects are generally not identified
## We standardize the solution in the following way :
## Mean within connected component of all fixed effects except the first
## is zero
##
## Unique solution with two components, not really with more
##
## Connected component : Breadth-first search
## components is an array of component
## A component is an array of set (length is number of values taken)
##
##############################################################################

"""
Solve a least square problem for a set of FixedEffects

`solve_coefficients!(y, fes, weights; method = :lsmr, maxiter = 10000, tol = 1e-8)`

### Arguments
* `y` : A `AbstractVector` 
* `fes`: A `Vector{<:FixedEffect}`
* `weights`: A `AbstractWeights`
* `method` : A `Symbol` for the method. Choices are :lsmr, :lsmr_threads, :lsmr_parallel, :qr and :cholesky
* `maxiter` : Maximum number of iterations
* `tol` : Tolerance


### Returns
* `b` : Solution of the least square problem
* `iterations`: Number of iterations
* `converged`: Did the algorithm converge?
Fixed effects are generally not identified. We standardize the solution 
in the following way: the mean of fixed effects within connected components is zero
(except for the first).

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)
solve_coefficients!(rand(10), [FixedEffect(p1), FixedEffect(p2)])
```
"""
function solve_coefficients!(y::AbstractVector, fes::Vector{<: FixedEffect}, weights::AbstractWeights  = Weights(Ones{eltype(y)}(length(y))); method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    any(ismissing.(fes)) && error("Some FixedEffect has a missing value for reference or interaction")
    feM = FixedEffectMatrix(fes, sqrt.(weights.values), Val{method})
    solve_coefficients!(y, feM; maxiter = maxiter, tol = tol)
end

function normalize!(x, b::AbstractVector, feM::AbstractFixedEffectMatrix; kwargs...)
    # The solution is generally not unique. Find connected components and scale accordingly
    findintercept = findall(fe -> isa(fe.interaction, Ones), fixedeffects(feM))
    if length(findintercept) >= 2
        components = connectedcomponent(view(fixedeffects(feM), findintercept))
        rescale!(x, feM, findintercept, components)
    end
    fes = fixedeffects(feM)
    newfes = [zeros(length(b)) for j in 1:length(fes)]
    for j in 1:length(fes)
        newfes[j] = x[j][fes[j].refs]
    end
    return newfes
end

function connectedcomponent(fes::AbstractVector{<:FixedEffect})
    # initialize
    where = initialize_where(fes)
    refs = initialize_refs(fes)
    nobs = size(refs, 2)
    visited = fill(false, nobs)
    components = Vector{Set{Int}}[]
    # start
    for i in 1:nobs
        if !visited[i]
            component = Set{Int}[Set{Int}() for fe in fes]
            connectedcomponent!(component, visited, i, refs, where)
            push!(components, component)
        end
    end
    return components
end

function initialize_where(fes::AbstractVector{<:FixedEffect})
    where = Vector{Set{Int}}[]
    for j in 1:length(fes)
        fe = fes[j]
        wherej = Set{Int}[Set{Int}() for i in 1:fe.n]
        for i in 1:length(fe.refs)
            push!(wherej[fe.refs[i]], i)
        end
        push!(where, wherej)
    end
    return where
end

function initialize_refs(fes::AbstractVector{<:FixedEffect})
    nobs = length(fes[1].refs)
    refs = fill(zero(Int), length(fes), nobs)
    for j in 1:length(fes)
        refs[j, :] = fes[j].refs
    end
    return refs
end

# Breadth-first search
function connectedcomponent!(component::Vector{Set{N}}, visited::Vector{Bool}, 
    i::Integer, refs::AbstractMatrix{N}, where::Vector{Vector{Set{N}}})  where {N}
    tovisit = Set{N}(i)
    while !isempty(tovisit)
        i = pop!(tovisit)
        visited[i] = true
        # for each fixed effect
        for j in 1:size(refs, 1)
            ref = refs[j, i]
            # if category has not been encountered
            if !(ref in component[j])
                # mark category as encountered
                push!(component[j], ref)
                # add other observations with same component in list to visit
                for k in where[j][ref]
                    if !visited[k]
                        push!(tovisit, k)
                    end
                end
            end
        end
    end
end

function rescale!(fev::Vector{Vector{T}}, feM::AbstractFixedEffectMatrix, 
                  findintercept,
                  components::Vector{Vector{Set{N}}}) where {T, N}
    fes = fixedeffects(feM)
    adj1 = zero(T)
    i1 = findintercept[1]
    for component in components
        for i in reverse(findintercept)
            # demean all fixed effects except the first
            if i != 1
                adji = zero(T)
                for j in component[i]
                    adji += fev[i][j]
                end
                adji = adji / length(component[i])
                for j in component[i]
                    fev[i][j] -= adji
                end
                adj1 += adji
            else
                # rescale the first fixed effects
                for j in component[i1]
                    fev[i1][j] += adj1
                end
            end
        end
    end
end
