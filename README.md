[![Build Status](https://travis-ci.org/matthieugomez/FixedEffects.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffects.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffects.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffects.jl?branch=master)

This package solves least squares problem with high dimensional fixed effects. i.e for a matrix `D` of high dimensional fixed effects, it finds `b` and `ϵ` such that `y = D'b + ϵ` with `E[Dϵ] = 0`. 

The package defines two functions `solve_coefficients`, that solves for the coefficients `b`, and `solve_residuals`, that solves for the residuals `ϵ`.

```julia
using FixedEffects
# define fixed effects:
p1 = FixedEffect(repeat(1:5, inner = 2))
# combine fixed effects
p2 = FixedEffect(repeat(1:2, outer = 5), repeat(1:2, inner = 5))
# define interacted fixed effects
p3 = FixedEffect(repeat(1:5, outer = 2), interaction = rand(10))

# partial out a vector
x = rand(10)
solve_residuals!(x, [p1, p2])

# partial out a matrix
X = rand(10, 5)
solve_residuals!(X, [p1, p2])

# find the fixed effect coefficients
solve_coefficients!(x, [p1, p3])
```

Use `?solve_residuals!` or `solve_coefficients!` to see all the possible syntax.

This package is the backend for the package [FixedEffectModels.jl](https://github.com/matthieugomez/FixedEffectModels.jl), that estimates more general linears model with high-dimensional fixed effect.

