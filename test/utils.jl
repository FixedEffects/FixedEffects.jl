using FixedEffects, Test

v1 = ["ok", missing, "ok", "first"]
v2 = [1, 2, 3, 1]
v3 = [missing, 1, 1, missing]

@test FixedEffect(v1).refs == [1, 0, 1, 2]
@test FixedEffect(v3).refs ==  [0, 1, 1, 0]
@test FixedEffect(v1, v2, v3).refs == [0, 0, 1, 0]




show(FixedEffect(v1, v2))
show(FixedEffect(v1; interaction = v2))
show(FixedEffect(v1, v3 ;interaction = v2))



