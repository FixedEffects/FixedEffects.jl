[![Build Status](https://travis-ci.org/matthieugomez/FixedEffects.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffects.jl)
[![pipeline status](https://gitlab.com/JuliaGPU/FixedEffects-jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/FixedEffects-jl/commits/master)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffects.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffects.jl?branch=master)

This package solves least squares problem with high dimensional fixed effects. For a matrix `D` of high dimensional fixed effects, it finds `b` and `ϵ` such that `y = D'b + ϵ` with `E[Dϵ] = 0`. 

It is the back end for the package [FixedEffectModels.jl](https://github.com/matthieugomez/FixedEffectModels.jl), that estimates more general linears model with high-dimensional fixed effect.

 The package defines two functions `solve_coefficients`, that returns the coefficients `b`, and `solve_residuals`, that returns the residuals `ϵ`. See `?solve_residuals!` or `?solve_coefficients!` for the full syntax.


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

# find the fixed effect coefficients
solve_coefficients!(x, [p1, p3])
```



## Installation
The package is registered in the [`General`](https://github.com/JuliaRegistries/General) registry and so can be installed at the REPL with `] add FixedEffects`.