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

@testset "koren_limiter" begin
    @test koren_limiter(0.0f0) == 0.0f0
    @test koren_limiter(1.0f0) == 1.0f0
    @test koren_limiter(2.0f0) ≈ 5.0f0 / 3.0f0

    @test koren_limiter(-1.0f0) == 0.0f0
    @test koren_limiter(3.0f0) == 2.0f0

    for r in (0.0f0, 1.0f0, 2.0f0)
        v = koren_limiter(r)
        @test 0.0f0 <= v <= 2.0f0
    end

    rs = Float32[-1, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    phis = koren_limiter.(rs)
    @test all(diff(phis) .>= 0.0f0)
end

@testset "compute_limited_flux_x" begin
    p = Params(Float32; Nx=6, Nz=4, H=4.0, dz0=1.0, z_fact=1.0)

    scalar_rho = zeros(Float32, p.ka, p.ia)
    rho = ones(Float32, p.ka, p.ia)
    rho_u = zeros(Float32, p.ka, p.ia)

    for j in 1:p.ia
        scalar_rho[p.ks, j] = j
    end

    k = p.ks
    i = p.is + 1

    # Positive rho_u -> upwind from left
    rho_u[k, i] = 1.0f0
    flux_pos = compute_limited_flux_x(scalar_rho, rho, rho_u, k, i, p)
    s_im1 = scalar_rho[k, i-1] / rho[k, i-1]
    s_i = scalar_rho[k, i] / rho[k, i]
    s_ip1 = scalar_rho[k, i+1] / rho[k, i+1]
    r_num = s_ip1 - s_i
    r_den = s_i - s_im1
    r = abs(r_den) < EPS ? 1.0f0 : r_num / r_den
    phi = koren_limiter(r)
    s_face_exp = s_i + 0.5f0 * phi * r_den
    @test flux_pos ≈ s_face_exp * rho_u[k, i]

    # Negative rho_u -> upwind from right
    rho_u[k, i] = -1.0f0
    flux_neg = compute_limited_flux_x(scalar_rho, rho, rho_u, k, i, p)
    s_ip2 = scalar_rho[k, i+2] / rho[k, i+2]
    r_num = s_i - s_ip1
    r_den = s_ip1 - s_ip2
    r = abs(r_den) < EPS ? 1.0f0 : r_num / r_den
    phi = koren_limiter(r)
    s_face_exp = s_ip1 + 0.5f0 * phi * r_den
    @test flux_neg ≈ s_face_exp * rho_u[k, i]

    # Zero rho_u -> centered interpolation
    rho_u[k, i] = 0.0f0
    flux_zero = compute_limited_flux_x(scalar_rho, rho, rho_u, k, i, p)
    s_face_h = 0.5f0 * (s_i + s_ip1)
    @test flux_zero == 0.0f0
    @test s_face_h ≈ f2h(s_i, s_ip1, p.dx, p.dx)
end

@testset "compute_limited_flux_z" begin
    # Create a short vertical column with simple properties
    p = Params(Nx=1, Nz=4, halo=0, Lx=4.0f0, H=4.0f0, dz0=1.0f0, z_fact=1.0f0,
               ns=1, tout_int=1.0)
    rho       = fill(1.0f0, p.Nz, p.Nx)
    scalar_rho = reshape(Float32.(1:p.Nz), p.Nz, p.Nx)
    rho_w     = zeros(Float32, p.Nz, p.Nx)

    dt = p.dt

    # --- Upward velocity at bottom boundary ---
    rho_w[1,1] = 0.5f0
    f = compute_limited_flux_z(scalar_rho, rho, rho_w, 1, 1, p)
    s_k   = scalar_rho[1,1] / rho[1,1]
    s_kp1 = scalar_rho[2,1] / rho[2,1]
    expect_face = f2h(s_k, s_kp1, p.dz[1], p.dz[2])
    expect_face = min(expect_face, scalar_rho[1,1] / (rho_w[1,1] * dt))
    @test f ≈ expect_face * rho_w[1,1]

    # --- Downward velocity near top boundary ---
    rho_w[3,1] = -0.5f0
    f = compute_limited_flux_z(scalar_rho, rho, rho_w, 3, 1, p)
    s_k   = scalar_rho[3,1] / rho[3,1]
    s_kp1 = scalar_rho[4,1] / rho[4,1]
    expect_face = f2h(s_k, s_kp1, p.dz[3], p.dz[4])
    expect_face = min(expect_face, scalar_rho[4,1] / (-rho_w[3,1] * dt))
    @test f ≈ expect_face * rho_w[3,1]

    # --- Zero velocity interior ---
    rho_w[2,1] = 0.0f0
    f = compute_limited_flux_z(scalar_rho, rho, rho_w, 2, 1, p)
    s_k   = scalar_rho[2,1] / rho[2,1]
    s_kp1 = scalar_rho[3,1] / rho[3,1]
    expect_face = f2h(s_k, s_kp1, p.dz[2], p.dz[3])
    @test f ≈ expect_face * rho_w[2,1]
end
