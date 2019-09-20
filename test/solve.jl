using FixedEffects, FillArrays

p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)



method_s = [:lsmr, :lsmr_threads, :lsmr_parallel]
if Base.USE_GPL_LIBS
	push!(method_s, :cholesky, :qr)
end

for method in method_s
	solve_coefficients!(x, [FixedEffect(p1), FixedEffect(p2)], method = method)
	solve_residuals!(x, [FixedEffect(p1), FixedEffect(p2)]; method = method)
	solve_residuals!([x x x x x], [FixedEffect(p1), FixedEffect(p2)]; method = method)
end


p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)
fes = [FixedEffect(p1, interaction = FillArrays.Ones{Float32}(length(p1))), FixedEffect(p2, interaction = FillArrays.Ones{Float32}(length(p2)))]
x = rand(Float32, 10)
solve_residuals!(x, fes; method = :lsmr)
