@setup_workload begin
    p1 = repeat(1:5, inner = 2)
    p2 = repeat(1:5, outer = 2)
    x = rand(10)
    fes = [FixedEffect(p1), FixedEffect(p2)]
    @compile_workload begin
        solve_residuals!(copy(x), fes)
        solve_residuals!(copy(x), fes, Weights(ones(10)))
        solve_coefficients!(copy(x), fes)
    end
end
