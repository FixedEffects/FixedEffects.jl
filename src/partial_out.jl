"""
Partial out a vector or a matrix

### Arguments
* `X` : a `AbstractVector{Float64}` or an `AbstractMatrix{Float64}`
* `fes`: A Vector of `FixedEffect`
* w: A Vector of weights
* `add_mean`. If true,  the mean of the initial variable is added to the residuals.
* `maxiter` : Maximum number of iterations
* `tol` : tolerance
* `method` : A symbol for the method. Default is :lsmr (akin to conjugate gradient descent). Other choices are :qr and :cholesky (factorization methods)


### Returns
* `X` : a residualized version of `X` 
* `iterations`: a vector of iterations for each column
* `converged`: a vector of success for each column

### Examples
```julia
using  FixedEffectModels
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
X = rand(10, 5)
partial_out!(X, [FixedEffect(p1), FixedEffect(p2)])
```
"""
function partial_out!(Y::Union{AbstractVector{Float64}, AbstractMatrix{Float64}}, fes::Vector{<: FixedEffect}; w = Ones{Float64}(size(Y, 1)), add_mean::Bool = false, maxiter::Integer = 10000, tol::Real = 1e-8, method::Symbol = :lsmr)

    sqrtw = sqrt.(w)
    fep = FixedEffectProblem(fes, sqrtw, Val{method})

    Y .= Y .* sqrtw
    if add_mean
        m = mean(Y, dims = 1)
    end

    iterations = Int[]
    converged = Bool[]
    partial_out!(Y, fep, iterations, converged; maxiter = maxiter, tol = tol)

    if add_mean
        Y .= Y .+ m
    end
    Y .= Y ./ sqrtw

    return Y, iterations, converged
end



"""
Find the Fixed Effect Coefficients, i.e. return c such that fes * c = b

### Arguments
* `b` : a `AbstractVector{Float64}` \
* `fes`: A Vector of `FixedEffect`

## Returns
* `c`: A vector such that fes * c = b

### Examples
```julia
using FixedEffectModels
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
X = rand(10)
getfe(b, [FixedEffect(p1), FixedEffect(p2)])
```
"""

function getfe(b::Union{AbstractVector{Float64}, AbstractMatrix{Float64}}, fes::Vector{<: FixedEffect}; maxiter::Integer = 10000, tol::Real = 1e-8, method::Symbol = :lsmr)
    fep = FixedEffectProblem(fes, Ones{Float64}(length(b)), Val{method})
    fev, iterations, converged = getfe!(fep, b)
    newfes = similar(fes)
    for j in 1:length(fes)
        newfes[j] = fev[j][fes[j].refs]
    end
end





