tests = ["types.jl", "solve.jl"]
println("Running tests:")

# A work around for tests to run on older versions of Julia
using Pkg
if VERSION >= v"1.8"
	Pkg.add("Metal")
	using Metal
end

using Test, StatsBase, CUDA, Metal, FixedEffects, PooledArrays, CategoricalArrays

for test in tests
	try
		include(test)
		println("\t\033[1m\033[32mPASSED\033[0m: $(test)")
	 catch e
	 	println("\t\033[1m\033[31mFAILED\033[0m: $(test)")
	 	showerror(stdout, e, backtrace())
	 	rethrow(e)
	 end
end