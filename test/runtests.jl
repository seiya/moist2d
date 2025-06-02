using Test
include("../moist.jl")

@testset "f2h" begin
    @test f2h(1.0f0, 2.0f0, 3.0f0, 4.0f0) ≈ (1.0f0*4.0f0 + 2.0f0*3.0f0) / 7.0f0
    # Identity: when both face values are equal the result should be the same
    @test f2h(5.0f0, 5.0f0, 2.0f0, 7.0f0) ≈ 5.0f0
    # Nonuniform dz weighting
    @test f2h(2.0f0, 4.0f0, 1.0f0, 3.0f0) ≈ (2.0f0*3.0f0 + 4.0f0*1.0f0) / 4.0f0
    # Symmetry when swapping vf and dz pairs
    @test f2h(2.0f0, 4.0f0, 1.0f0, 3.0f0) ≈ f2h(4.0f0, 2.0f0, 3.0f0, 1.0f0)
end

@testset "exchange_halo" begin
    p = Params(; Nx=4, Nz=2, halo=1, H=200.0f0, dz0=100.0f0)
    A = zeros(FT, p.Nz, p.ia)
    interior_vals = reshape(collect(1:p.Nz * p.Nx), p.Nz, p.Nx)
    A[:, p.is:p.ie] .= interior_vals
    exchange_halo!(A, p)
    @test A[:, 1:p.halo] == A[:, (p.ie-p.halo+1):p.ie]
    @test A[:, (p.ie+1):(p.ie+p.halo)] == A[:, p.is:(p.is+p.halo-1)]
end

@testset "solve_tridiagonal!" begin
    # Coefficients for a simple 3x3 tridiagonal system
    a = [0.0, -1.0, -1.0]
    b = [2.0, 2.0, 2.0]
    c = [-1.0, -1.0, 0.0]
    d = [1.0, 2.0, 3.0]

    expected = [2.5, 4.0, 3.5]

    ac = copy(a)
    bc = copy(b)
    cc = copy(c)
    dc = copy(d)

    solve_tridiagonal!(ac, bc, cc, dc)

    @test dc ≈ expected
end

@testset "saturation_vapor_pressure" begin
    T0 = FT(273.15)
    es, des_dT = saturation_vapor_pressure(T0)
    @test es ≈ ES0 atol = FT(1e-3)

    eps_T = eps(T0)
    es2, _ = saturation_vapor_pressure(T0 + eps_T)
    @test des_dT ≈ (es2 - es) / eps_T rtol = FT(5e-2)
end

@testset "saturation_specific_humidity" begin
    p = 1.0f5
    T_k = 300.0f0
    es, _ = saturation_vapor_pressure(T_k)
    tmp = p - (1.0f0 - EPSvap) * es
    qsat_ref = EPSvap * es / tmp
    dqsat_des_ref = EPSvap * p / tmp^2
    qsat, dqsat_des = saturation_specific_humidity(p, es)
    @test qsat ≈ qsat_ref

    delta = eps(es)
    qsat2, _ = saturation_specific_humidity(p, es + delta)
    dqsat_des_fd = (qsat2 - qsat) / delta
    @test isapprox(dqsat_des, dqsat_des_fd; atol=2f-6)
end
