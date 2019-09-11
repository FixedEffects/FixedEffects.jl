using FixedEffects

p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)
xs = rand(10, 5)

if Base.USE_GPL_LIBS
	method_s = [:cholesky, :qr, :lsmr, :lsmr_threads, :lsmr_parallel]
else
	method_s = [:lsmr, :lsmr_threads, :lsmr_parallel]
end

for method in method_s
	solve_residuals!(x, [FixedEffect(p1), FixedEffect(p2)]; method = method)
	solve_residuals!(xs, [FixedEffect(p1), FixedEffect(p2)]; method = method)
end
