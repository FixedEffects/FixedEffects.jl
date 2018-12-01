using FixedEffects

p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)

if Base.USE_GPL_LIBS
	method_s = [:cholesky, :qr, :lsmr, :lsmr_parallel, :lsmr_threads]
else
	method_s = [:lsmr, :lsmr_parallel, :lsmr_threads]
end
for method in method_s
	solve_coefficients!(x, [FixedEffect(p1), FixedEffect(p2)], method = method)
end

