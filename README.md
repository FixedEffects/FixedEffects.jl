[![Build Status](https://travis-ci.org/matthieugomez/FixedEffects.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffects.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffects.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffects.jl?branch=master)

This package solves least squares problem when the regressors are high dimensional fixedeffects. i.e for a matrix `D` of high dimensional fixed effefcts, it finds `b` and `ϵ` such that `y = D'b + ϵ`

The package defines two functions `solve_coefficients` (that solves for the coefficients `b`) and `solve_residuals` (that solves for the residuals `ϵ`).

```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10, 1)
solve_coefficients!(x, [FixedEffect(p1), FixedEffect(p2)])
# partial out a vector
solve_residuals!(x, [FixedEffect(p1), FixedEffect(p2)])
X = rand(10, 5)
# partial out a matrix
solve_residuals!(X, [FixedEffect(p1), FixedEffect(p2)])
```

This package is used as a backend for the package [FixedEffectModels.jl](https://github.com/matthieugomez/FixedEffectModels.jl), that estimates more general linears model with high dimensional fixed effects.


