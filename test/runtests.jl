using Test
include("../moist.jl")

@testset "f2h" begin
    @test f2h(1.0f0, 2.0f0, 3.0f0, 4.0f0) ≈ (1.0f0*4.0f0 + 2.0f0*3.0f0) / 7.0f0
end
