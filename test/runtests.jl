using Test

# Determine precision from command line argument (Float32 or Float64)
precision = Float32
if length(ARGS) > 0
    if ARGS[1] == "Float64"
        precision = Float64
    elseif ARGS[1] != "Float32"
        error("Unsupported precision " * ARGS[1])
    end
end

include("../moist.jl")

# Set precision for tests
set_precision!(precision)

# Set constants
set_constants!(FT)

# Helper for tolerances depending on precision
# When using Float64, the default tolerance scales with 1e-6
tol(f32, f64=f32 * 1e-6) = FT === Float64 ? FT(f64) : FT(f32)

@testset "f2h" begin
    @test f2h(FT(1.0), FT(2.0), FT(3.0), FT(4.0)) ≈ (FT(1.0)*FT(4.0) + FT(2.0)*FT(3.0)) / FT(7.0)
    # Identity: when both face values are equal the result should be the same
    @test f2h(FT(5.0), FT(5.0), FT(2.0), FT(7.0)) ≈ FT(5.0)
    # Nonuniform dz weighting
    @test f2h(FT(2.0), FT(4.0), FT(1.0), FT(3.0)) ≈ (FT(2.0)*FT(3.0) + FT(4.0)*FT(1.0)) / FT(4.0)
    # Symmetry when swapping vf and dz pairs
    @test f2h(FT(2.0), FT(4.0), FT(1.0), FT(3.0)) ≈ f2h(FT(4.0), FT(2.0), FT(3.0), FT(1.0))
end

@testset "exchange_halo" begin
    p = Params(; Nx=4, Nz=2, halo=1, H=FT(200.0), dz0=FT(100.0))
    A = zeros(FT, p.Nz, p.ia)
    interior_vals = reshape(collect(1:p.Nz * p.Nx), p.Nz, p.Nx)
    A[:, p.is:p.ie] .= interior_vals
    exchange_halo!(A, p)
    @test A[:, 1:p.halo] == A[:, (p.ie-p.halo+1):p.ie]
    @test A[:, (p.ie+1):(p.ie+p.halo)] == A[:, p.is:(p.is+p.halo-1)]
end

@testset "solve_tridiagonal!" begin
    # Coefficients for a simple 3x3 tridiagonal system
    a = FT[0.0, -1.0, -1.0]
    b = FT[2.0, 2.0, 2.0]
    c = FT[-1.0, -1.0, 0.0]
    d = FT[1.0, 2.0, 3.0]

    expected = FT[2.5, 4.0, 3.5]

    ac = copy(a)
    bc = copy(b)
    cc = copy(c)
    dc = copy(d)

    solve_tridiagonal!(ac, bc, cc, dc)

    @test dc ≈ expected
end

@testset "koren_limiter" begin
    @test koren_limiter(FT(0.0)) == FT(0.0)
    @test koren_limiter(FT(1.0)) == FT(1.0)
    @test koren_limiter(FT(2.0)) ≈ FT(5.0) / FT(3.0)

    @test koren_limiter(FT(-1.0)) == FT(0.0)
    @test koren_limiter(FT(3.0)) == FT(2.0)

    for r in (FT(0.0), FT(1.0), FT(2.0))
        v = koren_limiter(r)
        @test FT(0.0) <= v <= FT(2.0)
    end

    rs = FT[-1, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    phis = koren_limiter.(rs)
    @test all(diff(phis) .>= FT(0.0))
end

@testset "compute_limited_flux_x" begin
    p = Params(FT; Nx=6, Nz=4, H=FT(4.0), dz0=FT(1.0), z_fact=FT(1.0))

    scalar_rho = zeros(FT, p.ka, p.ia)
    rho = ones(FT, p.ka, p.ia)
    rho_u = zeros(FT, p.ka, p.ia)

    for j in 1:p.ia
        scalar_rho[p.ks, j] = j
    end

    k = p.ks
    i = p.is + 1

    # Positive rho_u -> upwind from left
    rho_u[k, i] = FT(1.0)
    flux_pos = compute_limited_flux_x(scalar_rho, rho, rho_u, k, i, p)
    s_im1 = scalar_rho[k, i-1] / rho[k, i-1]
    s_i = scalar_rho[k, i] / rho[k, i]
    s_ip1 = scalar_rho[k, i+1] / rho[k, i+1]
    r_num = s_ip1 - s_i
    r_den = s_i - s_im1
    r = abs(r_den) < EPS ? FT(1.0) : r_num / r_den
    phi = koren_limiter(r)
    s_face_exp = s_i + FT(0.5) * phi * r_den
    @test flux_pos ≈ s_face_exp * rho_u[k, i]

    # Negative rho_u -> upwind from right
    rho_u[k, i] = -FT(1.0)
    flux_neg = compute_limited_flux_x(scalar_rho, rho, rho_u, k, i, p)
    s_ip2 = scalar_rho[k, i+2] / rho[k, i+2]
    r_num = s_i - s_ip1
    r_den = s_ip1 - s_ip2
    r = abs(r_den) < EPS ? FT(1.0) : r_num / r_den
    phi = koren_limiter(r)
    s_face_exp = s_ip1 + FT(0.5) * phi * r_den
    @test flux_neg ≈ s_face_exp * rho_u[k, i]

    # Zero rho_u -> centered interpolation
    rho_u[k, i] = FT(0.0)
    flux_zero = compute_limited_flux_x(scalar_rho, rho, rho_u, k, i, p)
    s_face_h = FT(0.5) * (s_i + s_ip1)
    @test flux_zero == FT(0.0)
    @test s_face_h ≈ f2h(s_i, s_ip1, p.dx, p.dx)
end

@testset "compute_limited_flux_z" begin
    # Create a short vertical column with simple properties
    p = Params(Nx=1, Nz=4, halo=0, Lx=FT(4.0), H=FT(4.0), dz0=FT(1.0), z_fact=FT(1.0),
               ns=1, tout_int=1.0)
    rho       = fill(FT(1.0), p.Nz, p.Nx)
    scalar_rho = reshape(FT.(1:p.Nz), p.Nz, p.Nx)
    rho_w     = zeros(FT, p.Nz, p.Nx)

    dt = p.dt

    # --- Upward velocity at bottom boundary ---
    rho_w[1,1] = FT(0.5)
    f = compute_limited_flux_z(scalar_rho, rho, rho_w, 1, 1, p)
    s_k   = scalar_rho[1,1] / rho[1,1]
    s_kp1 = scalar_rho[2,1] / rho[2,1]
    expect_face = f2h(s_k, s_kp1, p.dz[1], p.dz[2])
    expect_face = min(expect_face, scalar_rho[1,1] / (rho_w[1,1] * dt))
    @test f ≈ expect_face * rho_w[1,1]

    # --- Downward velocity near top boundary ---
    rho_w[3,1] = -FT(0.5)
    f = compute_limited_flux_z(scalar_rho, rho, rho_w, 3, 1, p)
    s_k   = scalar_rho[3,1] / rho[3,1]
    s_kp1 = scalar_rho[4,1] / rho[4,1]
    expect_face = f2h(s_k, s_kp1, p.dz[3], p.dz[4])
    expect_face = min(expect_face, scalar_rho[4,1] / (-rho_w[3,1] * dt))
    @test f ≈ expect_face * rho_w[3,1]

    # --- Zero velocity interior ---
    rho_w[2,1] = FT(0.0)
    f = compute_limited_flux_z(scalar_rho, rho, rho_w, 2, 1, p)
    s_k   = scalar_rho[2,1] / rho[2,1]
    s_kp1 = scalar_rho[3,1] / rho[3,1]
    expect_face = f2h(s_k, s_kp1, p.dz[2], p.dz[3])
    @test f ≈ expect_face * rho_w[2,1]
end

@testset "implicit_correction" begin
    # Minimal parameters for a single column with Nz=3
    p = Params(Nz=3, Nx=1, H=FT(3.0), Lx=FT(1.0), dz0=FT(1.0), z_fact=FT(1.0),
               beta_offcenter=FT(1.0))

    # Constant input arrays
    rho_w = fill(FT(0.1), p.ka)
    drho_w = fill(FT(0.05), p.ka)
    theta = fill(FT(300.0), p.ka)
    rt2pres = fill(FT(1.0), p.ka)
    dt = FT(0.1)

    # Save the initial state of rho_w for constructing the system later
    rho_w_initial = copy(rho_w)

    # Apply the implicit correction
    implicit_correction!(rho_w, drho_w, theta, rt2pres, dt, p)

    # Rebuild the tridiagonal system using the initial rho_w
    ks, ke = p.ks, p.ke
    theta_h = similar(theta)
    for k in ks:ke-1
        theta_h[k] = f2h(theta[k], theta[k+1], p.dz[k], p.dz[k+1])
    end

    fact_tp = FT(0.5) * (FT(1.0) + p.beta_offcenter)
    fact_tm = FT(0.5) * (FT(1.0) - p.beta_offcenter)
    dt_tp2 = (dt * fact_tp)^2

    a = zeros(FT, p.Nz - 1)
    b = zeros(FT, p.Nz - 1)
    c = zeros(FT, p.Nz - 1)
    d = zeros(FT, p.Nz - 1)
    for k in ks:ke-1
        idx = k - ks + 1
        b[idx] = dt_tp2 * (rt2pres[k] / p.dz[k] + rt2pres[k+1] / p.dz[k+1]) *
                 theta_h[k] / (p.z[k+1] - p.z[k])
        if k > ks
            a[idx] = -dt_tp2 * (rt2pres[k] * theta_h[k-1] /
                                ((p.z[k+1] - p.z[k]) * p.dz[k]) -
                                GRAV / (p.dz[k] + p.dz[k+1]))
        end
        if k < ke - 1
            c[idx] = -dt_tp2 * (rt2pres[k+1] * theta_h[k+1] /
                                ((p.z[k+1] - p.z[k]) * p.dz[k+1]) +
                                GRAV / (p.dz[k] + p.dz[k+1]))
        end
        d[idx] = rho_w_initial[k] + drho_w[k]
    end

    fact = fact_tm / fact_tp
    for k in ks:ke-1
        idx = k - ks + 1
        if k > ks
            d[idx] -= a[idx] * fact * rho_w_initial[k]
        end
        d[idx] -= b[idx] * fact * rho_w_initial[k]
        b[idx] += FT(1.0)
        if k < ke - 1
            d[idx] -= c[idx] * fact * rho_w_initial[k+1]
        end
    end

    # Verify that the updated rho_w satisfies A * rho_w = d
    for k in ks:ke-1
        idx = k - ks + 1
        lhs = b[idx] * rho_w[k]
        if k > ks
            lhs += a[idx] * rho_w[k-1]
        end
        if k < ke - 1
            lhs += c[idx] * rho_w[k+1]
        end
        @test lhs ≈ d[idx] atol=tol(1e-6)
    end
end

@testset "compute_step mass" begin
    p = Params(Nx=2, Nz=2, Lx=FT(200.0), H=FT(210.0), dz0=FT(100.0),
                z_fact=FT(1.1), ns=1, nu=FT(0.0), kappa=FT(0.0))
    s = allocate_state(p)
    fill!(s.rho, FT(1.0))
    fill!(s.rho_u, FT(0.0))
    fill!(s.rho_w, FT(0.0))
    fill!(s.rho_theta, FT(1.0))
    fill!(s.rho_qv, FT(0.0))
    fill!(s.rho_qc, FT(0.0))
    fill!(s.rho_qr, FT(0.0))
    fill!(s.theta_ref, FT(1.0))
    mass_before = sum(s.rho[p.ks:p.ke, p.is:p.ie] .* p.dz[p.ks:p.ke])
    d_phys = fill(FT(0.01), p.ka, p.ia)
    zeros2d = zeros(FT, p.ka, p.ia)
    s0 = deepcopy(s)
    compute_step!(d_phys, zeros2d, zeros2d, zeros2d, zeros2d, zeros2d, s, s0, p, FT(1.0), 1)
    mass_after = sum(s.rho[p.ks:p.ke, p.is:p.ie] .* p.dz[p.ks:p.ke])
    expected = mass_before + FT(1.0) * sum(d_phys[p.ks:p.ke, p.is:p.ie] .* p.dz[p.ks:p.ke])
    @test isapprox(mass_after, expected; atol=tol(1e-6))
end

@testset "compute_step monotonicity" begin
    p = Params(Nx=4, Nz=2, Lx=FT(4.0), H=FT(2.0), dz0=FT(1.0),
                z_fact=FT(1.0), ns=1, nu=FT(0.0), kappa=FT(0.0), gamma_d=FT(0.0))
    s = allocate_state(p)
    fill!(s.rho, FT(1.0))
    fill!(s.rho_theta, FT(300.0))
    fill!(s.rho_w, FT(0.0))

    # Sinusoidal horizontal velocity
    amp = FT(0.1)
    base = FT(1.0)
    for i in p.is:p.ie
        u = base + amp * sin(4 * pi * (i - p.is) / p.Nx)
        s.rho_u[p.ks, i] = u * s.rho[p.ks, i]
    end
    exchange_halo!(s.rho_u, p)

    # Sinusoidal mixing ratio with small amplitude
    amp = FT(0.004)
    base = FT(0.005)
    for i in p.is:p.ie
        qv = base + amp * sin(2 * pi * (i - p.is) / p.Nx)
        s.rho_qv[p.ks, i] = qv * s.rho[p.ks, i]
    end
    exchange_halo!(s.rho_qv, p)

    s0 = deepcopy(s)
    zeros2d = zeros(FT, p.ka, p.ia)
    dt = p.dt
    compute_step!(zeros2d, zeros2d, zeros2d, zeros2d, zeros2d, zeros2d,
                  s, s0, p, dt, 1)

    qv_init = s0.rho_qv[p.ks, p.is:p.ie] ./ s0.rho[p.ks, p.is:p.ie]
    qv_new = s.rho_qv[p.ks, p.is:p.ie] ./ s.rho[p.ks, p.is:p.ie]

    @test minimum(qv_new) ≥ minimum(qv_init) - tol(1e-6)
    @test maximum(qv_new) ≤ maximum(qv_init) + tol(2e-6, 2e-6)
end

@testset "compute_step CwC" begin
    p = Params(Nx=4, Nz=2, Lx=FT(4.0), H=FT(2.0), dz0=FT(1.0),
                z_fact=FT(1.0), ns=1, nu=FT(0.0), kappa=FT(0.0), gamma_d=FT(0.0))
    s = allocate_state(p)
    fill!(s.rho_theta, FT(300.0))
    fill!(s.rho_w, FT(0.0))

    # Sinusoidal density
    amp = FT(0.1)
    base = FT(1.0)
    for i in p.is:p.ie
        dens = base + amp * sin(2 * pi * (i - p.is) / p.Nx)
        s.rho[p.ks, i] = dens
        dens = base + amp * sin(4 * pi * (i - p.is) / p.Nx)
        s.rho[p.ks+1, i] = dens
    end
    exchange_halo!(s.rho, p)

    # Sinusoidal horizontal velocity
    amp = FT(0.1)
    base = FT(1.0)
    for i in p.is:p.ie
        u = base + amp * sin(4 * pi * (i - p.is) / p.Nx)
        s.rho_u[p.ks, i] = u * s.rho[p.ks, i]
        s.rho_u[p.ks+1, i] = u * s.rho[p.ks+1, i]
    end
    exchange_halo!(s.rho_u, p)

    qv_const = FT(0.005)
    for i in p.is:p.ie, k in p.ks:p.ke
        s.rho_qv[k, i] = qv_const * s.rho[k, i]
    end
    exchange_halo!(s.rho_qv, p)

    s0 = deepcopy(s)
    zeros2d = zeros(FT, p.ka, p.ia)
    dt = p.dt
    compute_step!(zeros2d, zeros2d, zeros2d, zeros2d, zeros2d, zeros2d,
                  s, s0, p, dt, 1)

    qv_new1 = s.rho_qv[p.ks, p.is:p.ie] ./ s.rho[p.ks, p.is:p.ie]
    qv_new2 = s.rho_qv[p.ks+1, p.is:p.ie] ./ s.rho[p.ks+1, p.is:p.ie]
    @test maximum( abs.(qv_new1 .- qv_const) ) / qv_const <= tol(1e-6, 5e-7)
    @test maximum( abs.(qv_new2 .- qv_const) ) / qv_const <= tol(1e-6, 5e-7)
end

@testset "rk3_step" begin
    p = Params(Nx=4, Nz=4, H=FT(430.0), dz0=FT(100.0), z_fact=FT(1.05), ns=1)
    rngs = [Random.Xoshiro(i) for i in 1:Threads.nthreads()]
    s_init = init_state!(allocate_state(p), p, rngs)

    # Predict state after a single compute_step!
    s_single = deepcopy(s_init)
    zeros2d = zeros(FT, p.ka, p.ia)
    phys_args = (zeros2d, zeros2d, zeros2d, zeros2d, zeros2d, zeros2d)
    compute_step!(phys_args..., s_single, deepcopy(s_init), p, p.dt, p.ns)
    mass_pred = sum(s_single.rho)

    # Run one RK3 step
    s_rk3 = deepcopy(s_init)
    rk3_step!(s_rk3, p)
    mass_rk3 = sum(s_rk3.rho)

    arrays_finite = all(isfinite, s_rk3.rho) &&
                    all(isfinite, s_rk3.rho_u) &&
                    all(isfinite, s_rk3.rho_w) &&
                    all(isfinite, s_rk3.rho_theta) &&
                    all(isfinite, s_rk3.rho_qv) &&
                    all(isfinite, s_rk3.rho_qc) &&
                    all(isfinite, s_rk3.rho_qr)
    @test arrays_finite
    @test isapprox(mass_rk3, mass_pred; atol=tol(1e-6), rtol=tol(1e-6))
end

@testset "saturation_vapor_pressure" begin
    T0 = FT(273.15)
    es, des_dT = saturation_vapor_pressure(T0)
    @test es ≈ ES0 atol = tol(1e-6)

    eps_T = sqrt(eps(T0))
    es2, _ = saturation_vapor_pressure(T0 + eps_T)
    @test des_dT ≈ (es2 - es) / eps_T rtol = tol(2e-4, 2e-8)
end

@testset "saturation_specific_humidity" begin
    p = FT(1e5)
    T_k = FT(300.0)
    es, _ = saturation_vapor_pressure(T_k)
    tmp = p - (FT(1.0) - EPSvap) * es
    qsat_ref = EPSvap * es / tmp
    dqsat_des_ref = EPSvap * p / tmp^2
    qsat, dqsat_des = saturation_specific_humidity(p, es)
    @test qsat ≈ qsat_ref

    delta = sqrt(eps(es))
    qsat2, _ = saturation_specific_humidity(p, es + delta)
    dqsat_des_fd = (qsat2 - qsat) / delta
    @test dqsat_des ≈ dqsat_des_fd atol=tol(1e-6, 1e-11)
end

@testset "invert_T_for_thetae" begin
    qt = FT(0.02)
    theta_e = FT(320.0)
    T_est, qsat = invert_T_for_thetae(qt, theta_e, P0)
    @test 250 <= T_est <= 350
    @test qsat > 0

    es, _ = saturation_vapor_pressure(T_est)
    qsat_ref, _ = saturation_specific_humidity(P0, es)
    qd = FT(1.0) - qt
    cp = Cpd * qd + Cpl * qt
    p_d = P0 - es
    theta_e_ref = T_est * (p_d / P0)^(-Rd / cp) * exp(Lv * qsat_ref / (cp * T_est))
    @test theta_e_ref ≈ theta_e atol=tol(1e-4, 1e-7)
end

@testset "cloud_microphysics conservation" begin
    p = Params{FT}(; Nx=1, Nz=3, halo=0, dt=FT(1.0))
    rho = fill(FT(1.0), p.ka, p.ia)
    rho_theta = fill(FT(300.0), p.ka, p.ia)
    rho_qv = fill(FT(0.025), p.ka, p.ia)
    rho_qc = fill(FT(0.0), p.ka, p.ia)
    rho_qr = fill(FT(0.0), p.ka, p.ia)

    d_qv = zeros(FT, p.ka, p.ia)
    d_qc = zeros(FT, p.ka, p.ia)
    d_qr = zeros(FT, p.ka, p.ia)
    d_theta = zeros(FT, p.ka, p.ia)

    cloud_microphysics!(d_qv, d_qc, d_qr, d_theta,
        rho, rho_theta, rho_qv, rho_qc, rho_qr, p)

    rho_qv_new = rho_qv .+ d_qv .* p.dt
    rho_qc_new = rho_qc .+ d_qc .* p.dt
    rho_qr_new = rho_qr .+ d_qr .* p.dt

    mass_before = rho_qv .+ rho_qc .+ rho_qr
    mass_after = rho_qv_new .+ rho_qc_new .+ rho_qr_new

    @test maximum( abs.(mass_after .- mass_before) ) < tol(1e-10)
    @test all(rho_qv_new .>= 0) && all(rho_qc_new .>= 0) && all(rho_qr_new .>= 0)
end
