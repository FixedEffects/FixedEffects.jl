"""
Solve a least square problem for a set of FixedEffects

### Arguments
* `y` : an `AbstractVector{Float64}` or an `AbstractMatrix{Float64}`
* `fes`: A Vector of `FixedEffect`
* `maxiter` : Maximum number of iterations
* `w`: Weights
* `tol` : tolerance
* `method` : A symbol for the method. Default is :lsmr (akin to conjugate gradient descent). Other choices are :qr and :cholesky (factorization methods)


### Returns
* `res` : the residual of the least square problem
* `iterations`: number of iterations
* `converged`: did the algorithm converge?

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
X = rand(10, 5)
solve_residuals!(x, [FixedEffect(p1), FixedEffect(p2)])
```
"""

function solve_residuals!(y::AbstractVector{Float64}, fes::Vector{<: FixedEffect}; w = Ones{Float64}(length(y)), maxiter::Integer = 10000, tol::Real = 1e-8, method::Symbol = :lsmr)
    sqrtw = sqrt.(w)
    y .= y .* sqrtw
    fep = FixedEffectProblem(fes, sqrtw, Val{method})

    y, iteration, converged = solve_residuals!(y, fep; maxiter = maxiter, tol = tol)
    y .= y ./ sqrtw
    return y, iteration, converged
end

function solve_residuals!(y::AbstractMatrix{Float64}, fes::Vector{<: FixedEffect}; w = Ones{Float64}(size(y, 1)), maxiter::Integer = 10000, tol::Real = 1e-8, method::Symbol = :lsmr)
    sqrtw = sqrt.(w)
    y .= y .* sqrtw
    fep = FixedEffectProblem(fes, sqrtw, Val{method})

    y, iterations, convergeds = solve_residuals!(y, fep; maxiter = maxiter, tol = tol)
    y .= y ./ sqrtw
    return y, iterations, convergeds
end


"""
Solve a least square problem for a set of FixedEffects

### Arguments
* `y` : an `AbstractVector{Float64}` 
* `fes`: A Vector of `FixedEffect`
* `maxiter` : Maximum number of iterations
* `w`: Weights
* `tol` : tolerance
* `method` : A symbol for the method. Default is :lsmr (akin to conjugate gradient descent). Other choices are :qr and :cholesky (factorization methods)


### Returns
* `b` : the solution of the least square problem
* `iterations`: number of iterations
* `converged`: did the algorithm converge?

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
X = rand(10, 5)
solve_coefficients!(x, [FixedEffect(p1), FixedEffect(p2)])
```
"""
function solve_coefficients!(y, fes::Vector{<: FixedEffect}; w = Ones{Float64}(length(y)), maxiter::Integer = 10000, tol::Real = 1e-8, method::Symbol = :lsmr)

    sqrtw = sqrt.(w)
    fep = FixedEffectProblem(fes, sqrtw, Val{method})

    y .= y .* sqrtw
    fev, iteration, converged = solve_coefficients!(y, fep; maxiter = maxiter, tol = tol)
    newfes = [zeros(length(y)) for j in 1:length(fes)]
    for j in 1:length(fes)
        newfes[j] = fev[j][fes[j].refs]
    end
    return newfes, iteration, converged
end



