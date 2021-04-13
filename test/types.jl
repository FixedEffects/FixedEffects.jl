using Test
using FixedEffects
using FixedEffects: GroupedArray, group, factorize!
using StatsBase
using PooledArrays, CategoricalArrays
import Base: ==

==(x::FixedEffect{R,I}, y::FixedEffect{R,I}) where {R,I} =
    x.refs == y.refs && x.interaction == y.interaction && x.n == y.n

==(g1::GroupedArray{N}, g2::GroupedArray{N}) where N =
    g1.refs == g2.refs && g1.n == g2.n


@testset "FixedEffect" begin
    fe1 = FixedEffect(1:10)
    @test sprint(show, fe1) == "Fixed Effects"
    if VERSION <= v"1.5"
        @test sprint(show, MIME("text/plain"), fe1) == """
            Fixed Effects:
              refs (10-element Array{UInt32,1}):
                [1, 2, 3, 4, 5, ... ]
              interaction (UnitWeights):
                none"""
    else
        @test sprint(show, MIME("text/plain"), fe1) == """
            Fixed Effects:
              refs (10-element Vector{UInt32}):
                [1, 2, 3, 4, 5, ... ]
              interaction (UnitWeights):
                none"""
    end
    fe2 = FixedEffect(1:10, interaction=fill(1.23456789, 10))
    if VERSION <= v"1.5"
        @test sprint(show, MIME("text/plain"), fe2) == """
            Fixed Effects:
              refs (10-element Array{UInt32,1}):
                [1, 2, 3, 4, 5, ... ]
              interaction (10-element Array{Float64,1}):
                [1.23457, 1.23457, 1.23457, 1.23457, 1.23457, ... ]"""
    else
        @test sprint(show, MIME("text/plain"), fe2) == """
        Fixed Effects:
          refs (10-element Vector{UInt32}):
            [1, 2, 3, 4, 5, ... ]
          interaction (10-element Vector{Float64}):
            [1.23457, 1.23457, 1.23457, 1.23457, 1.23457, ... ]"""
    end

    @test_throws DimensionMismatch FixedEffect(1:10, interaction=fill(1, 5))

    @test size(fe1) == (10,)
    @test length(fe1) == 10
    @test eltype(fe1) == Int
    @test eltype(fe2) == Float64

    @test fe1[:] === fe1
    subfe1 = fe1[[1,2]]
    @test subfe1.refs == fe1.refs[1:2]
    @test subfe1.interaction == uweights(2)
    @test subfe1 == fe1[1:2]
    @test subfe1 == fe1[fe1.refs.<=2]
    @test_throws MethodError fe1[[1 2]]

    subfe2 = fe2[[1,2]]
    @test subfe2.refs == fe2.refs[1:2]
    @test subfe2.interaction == fe2.interaction[1:2]
    @test subfe2 == fe2[1:2]
    @test subfe2 == fe2[fe2.refs.<=2]
    @test_throws MethodError fe2[[1 2]]
end

@testset "GroupedArray" begin
    N = 10
    a1 = collect(1:N)
    g1 = group(a1)
    @test g1 == GroupedArray(collect(UInt32, 1:N), N)
    @test size(g1) == (N,)
    @test length(g1) == N
    @test g1[1] == UInt32(1)
    @test g1[1:2] == [UInt32(1), UInt32(2)]
    @test g1[g1.<=2] == g1[[1,2]] == g1[1:2]

    a2 = [1,2]
    @test_throws DimensionMismatch group(a1, a2)

    a = rand(N)
    g = group(a)
    @test factorize!(g) == g



    g = [0, 1, 2, 3, 1, 2, 0]
    @test group(g) == group(categorical(g))
    @test group(g) == group(PooledArray(g))

    g = [missing, 1, 2, 3, 1, 2, missing]
    @test group(g) == group(categorical(g))
    @test group(g) == group(PooledArray(g))

    g = [missing, "a", "b", "c", "a", "a", "a"]
    @test group(g) == group(categorical(g))
    @test group(g) == group(PooledArray(g))
end
