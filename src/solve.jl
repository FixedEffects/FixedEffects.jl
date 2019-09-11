
# this type must defined solve_residuals!, solve_coefficients!
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
function solve_residuals!(y::Union{AbstractVector, AbstractMatrix}, fes::AbstractVector{<: FixedEffect}, weights::AbstractWeights = Weights(Ones{eltype(y)}(size(y, 1))); method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    any(ismissing.(fes)) && error("Some FixedEffect has a missing value for reference or interaction")
    feM = FixedEffectMatrix(fes, sqrt.(weights.values), Val{method})
    y, iteration, converged = solve_residuals!(y, feM; maxiter = maxiter, tol = tol)
end

function solve_residuals!(X::AbstractMatrix, feM::AbstractFixedEffectMatrix; kwargs...)
    iterations = Int[]
    convergeds = Bool[]
    for x in eachcol(X)
        _, iteration, converged = solve_residuals!(x, feM; kwargs...)
        push!(iterations, iteration)
        push!(convergeds, converged)
    end
    return X, iterations, convergeds
end

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
Fixed effects are generally not unique. We standardize the solution 
in the following way: the mean of fixed effects within connected components is zero
(except for the first).
This gives the unique solution in the case of two fixed effects.

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)
solve_coefficients!(rand(10), [FixedEffect(p1), FixedEffect(p2)])
```
"""
function solve_coefficients!(y::AbstractVector, fes::AbstractVector{<: FixedEffect}, weights::AbstractWeights  = Weights(Ones{eltype(y)}(length(y))); method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    any(ismissing.(fes)) && error("Some FixedEffect has a missing value for reference or interaction")
    feM = FixedEffectMatrix(fes, sqrt.(weights.values), Val{method})
    solve_coefficients!(y, feM; maxiter = maxiter, tol = tol)
end

##############################################################################
##
## normalize!
## 
##############################################################################

function normalize!(x, b::AbstractVector, fes::AbstractVector{<:FixedEffect}; kwargs...)
    # The solution is generally not unique. Find connected components and scale accordingly
    idx_intercept = findall(fe -> isa(fe.interaction, Ones), fes)
    if length(idx_intercept) >= 2
        components = connectedcomponent(view(fes, idx_intercept))
        rescale!(x, fes, idx_intercept, components)
    end
    [x[j][fes[j].refs] for j in 1:length(fes)]
end


## Connected component : Breadth-first search
## components is an array of component
## A component is an array of set (length is number of values taken)
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

function rescale!(fev::Vector{Vector{T}}, fes::AbstractVector{<:FixedEffect}, 
                  idx_intercept,
                  components::Vector{Vector{Set{N}}}) where {T, N}
    adj1 = zero(T)
    i1 = idx_intercept[1]
    for component in components
        for i in reverse(idx_intercept)
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