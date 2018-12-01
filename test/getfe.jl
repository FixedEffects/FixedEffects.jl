using FixedEffects

p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
b = rand(10)
getfe!(b, [FixedEffect(p1), FixedEffect(p2)])