using FixedEffects, CategoricalArrays, Test

v1 = ["ok", missing, "ok", "first"]
v2 = [1, 2, 3, 1]
v3 = [missing, 1, 1, missing]

@test levels(group(v1)) == [1, 2]
@test group(v2) == categorical(v2)
@test levels(group(v3)) ==  [1]
@test levels(group(v1, v2, v3)) == [1]



show(FixedEffect(v1, v2))
show(FixedEffect(v1; interaction = v2))
show(FixedEffect(v1, v3 ;interaction = v2))


# test different syntaxes
v1 = categorical(collect(1:1000))
v2 = categorical(fill(1, 1000))
@test group(v1, v2) == collect(1:1000)
@test group(v1, v2) == collect(1:1000)
@test group(v1, v2) == collect(1:1000)
