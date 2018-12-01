using FixedEffects

p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
X = rand(10, 5)
partial_out!(X, [FixedEffect(p1), FixedEffect(p2)])
