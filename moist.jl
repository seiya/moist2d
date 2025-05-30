# moist.jl
# -------------------------------------------------------------
# 2-D moist convection on Arakawa C-grid with Kessler microphysics
# -------------------------------------------------------------

#using CUDA
using Logging, Printf
using LoopVectorization
using NCDatasets
using Parameters
using Random

const FT = Float32
#const FT = Float64

# ============================================================
# 0. Physical Constants
# ============================================================
const GRAV = FT(9.81)     # Acceleration due to gravity (m/s^2)
const Rd = FT(287.04)     # Gas constant for dry air (J/kg/K)
const Rv = FT(461.5)      # Gas constant for water vapor (J/kg/K)
const Cpd = FT(1004.64)   # Specific heat of dry air at constant pressure (J/kg/K)
const Cpv = FT(1846.0)    # Specific heat of water vapor at constant pressure (J/kg/K)
const Cpl = FT(4218.0)    # Specific heat of liquid water
const Cvd = Cpd - Rd        # Specific heat of dry air at constant volume (J/kg/K)
const P0 = FT(1.e5)       # Reference pressure (Pa)
const Lv = FT(2.501e6)      # Latent heat of vaporization (J/kg)
const Ka = FT(0.4)        # Kalman constant
const SIGMA = FT(5.67e-8) # Stefan-Boltzmann constant (W/m^2/K^4)

const EPSvap = Rd / Rv
const EPSTvap = Rv / Rd - FT(1.0)

# --- Kessler Microphysics Parameters ---
const K_auto = FT(1e-3)   # Autoconversion rate coefficient (s^-1)
const Qc_threshold = FT(1e-3) # Autoconversion threshold (kg/kg)
const K_accr = FT(2.2)    # Accretion rate coefficient (m^3/kg/s adjusted for typical rho_air)
const K_evap = FT(1e-2)   # Rain evaporation rate coefficient (-) - Needs tuning
const A_term = FT(36.3)   # Terminal velocity coefficient (m/s * (m^3/kg)^-0.1346) - Marshall-Palmer
const B_term = FT(0.1346) # Terminal velocity exponent (-) - Marshall-Palmer

# --- Saturation Vapor Pressure Calculation (Tetens' formula) ---
const T0 = FT(273.15)     # Reference temperature (K)
const ES0 = FT(610.78)     # Saturation vapor pressure at T0 (Pa)
const A_SAT = FT(17.67)   # Coefficient for saturation vapor pressure
const B_SAT = FT(243.5)   # Coefficient for saturation vapor pressure (deg C)

const EPS = FT(1e-12) # Small value to avoid division by zero

# ============================================================
# 1. Grid & parameters
# ============================================================
@with_kw struct Params{T}
    Lx::T = FT(320e3)    # domain length (m)
    H::T = FT(25e3)     # domain height (m)
    #Lx::T = FT(20e3)    # domain length (m)
    #H::T = FT(10e3)     # domain height (m)
    Nx::Int = 640
    #Nz::Int = 128       # active cells in x, z
    Nz::Int = 250
    #Nx::Int = 128
    #Nz::Int = 64
    #Nx::Int = 200
    #Nz::Int = 100
    halo::Int = 2       # halo size
    ia::Int = Nx + 2 * halo # total array size
    ka::Int = Nz
    is::Int = 1 + halo
    ie::Int = Nx + halo # halo indices
    ks::Int = 1
    ke::Int = Nz        # halo indices
    dx::T = Lx / Nx     # horizontal grid spacing (m)
    #dz0::T = T(50.0)    # depth of lowermost layer (m)
    dz0::T = T(100.0)
    #z_fact::T = T(1.02) # strech factor for z
    z_fact::T = T(1.0)
    #dz::T = H / Nz      # vertical grid spacing (m)
    #x::StepRangeLen{T} = StepRangeLen(dx * (T(0.5) - halo), dx, Nx + 2 * halo) # Cell centers x
    #z::StepRangeLen{T} = StepRangeLen(dz / T(2.0), dz, Nz)          # Cell centers z
    nu::T = T(1.0) # Viscosity (m2/s)
    #nu::T = T(0.0)
    Pr::T = T(0.7) # Prandtl number
    kappa::T = nu / Pr # Diffusivity (m2/s)
    gamma_d::T = T(0.1) # Divergence damping coefficient (1)
    #gamma_d::T = T(0.0)
    #beta_offcenter::T = T(0.1)   # Off-centering beta parameter
    beta_offcenter::T = T(1.0)
    #zf_sponge::T = T(0.2) # fact from the top of sponge
    zf_sponge::T = T(0.0)
    #do_mp = true # Cloud microphysics
    do_mp = false
    #do_auto = true # Autoconversion
    do_auto = false
    #do_accr = true # Accretion
    do_accr = false
    #do_revap = true # Rain evaporation
    do_revap = false
    #do_sed = true # Rain sedimentation
    do_sed = false
    #do_rd = true # Radiation
    do_rd = false
    #do_sf = true # Surface fluxes
    do_sf = false
    Cd::T = T(1.2e-3) # Bulk coefficient for momentum
    Ch::T = T(0.8e-3) # Bulk coefficient for heat
    Ce::T = T(1.0e-3) # Bulk coefficient for vapor
    uabs_min::T = T(5.0) # Minimum velocity for surface fluxes (m/s)
    temp_surf::T = T(300.0) # Surface temperature (K)
    k_abs_qv::T = T(0.15) # Absorption coefficient for vapor (m^2/kg)
    k_abs_qc::T = T(20.0) # Absorption coefficient for cloud water (m^2/kg)
    k_abs_qr::T = T(10.0) # Absorption coefficient for rain water (m^2/kg)
    surf_emissivity::T = T(1.0) # Surface emissivity
    tau_cooling::T = T(864000.0) # Timescale for Newtonian cooling in stratosphere (K/s)
    #u0::T = T(10.0) # Base horizontal velocity
    u0::T = T(0.0)
    theta0::T = T(300.0) # Base potential temperature (K)
    bndlayer_depth::T = T(1000.0) # Depth of boundary layer (m)
    z_tropopose::T = T(15e3) # Tropopause height (m)
    #dtheta_dz_trop::T = T(4.0e-3) # Vertical potential temperature gradient in the troposphere (K/m)
    dtheta_dz_trop::T = T(0.0)
    #dtheta_dz_stra::T = T(18.3e-3) # Vertical potential temperature gradient in the stratosphere (K/m)
    dtheta_dz_stra::T = T(0.0)
    #theta_pert = T(2.0) # Perturbation amplitude
    #theta_pert = T(1.0)
    #theta_pert = T(0.1)
    #theta_pert = T(0.01)
    theta_pert = T(0.0)
    #rel_hum_surf::T = T(90.0)    # Base surface relative humidity (%)
    #rel_hum_surf::T = T(100.0)
    rel_hum_surf::T = T(0.0)
    rel_hum_decay::T = T(4000.0) # e-folding height for relative humidity (m)
    #rel_hum_decay::T = T(0.0)
    #t_end::Float64 = 86400.0 * 10.0
    #t_end::Float64 = 3600.0 * 30.0        # total sim time (s)
    #t_end::Float64 = 3600.0
    #t_end::Float64 = 1800.0
    #t_end::Float64 = 1000.0
    #t_end::Float64 = 600.0
    #t_end::Float64 = 300.0
    #t_end::Float64 = 160.0
    t_end::Float64 = 0.5
    #tout_int::Float64 = 3600.0
    #tout_int::Float64 = 300.0     # time interval for output (s)
    #tout_int::Float64 = 100.0
    #tout_int::Float64 = 30.0
    #tout_int::Float64 = 10.0
    #tout_int::Float64 = 5.0
    #tout_int::Float64 = 1.0
    tout_int::Float64 = 0.5
    ns::Int = 3                   # Number of short steps
    #ns::Int = 1
    dt::T = T(0.0)               # time step (s) - Calculated below
    N_steps::Int = 0              # Calculated below
    Nint_steps::Int = 0           # Calculated below
    x::Vector{T} = []
    zf::Vector{T} = []
    dz::Vector{T} = []
    z::Vector{T} = []
end

function Params(T=FT; kwargs...)
    p_initial = Params{T}(; kwargs...) # Create with defaults or user kwargs

    # Time step
    #CFL = 0.8
    #CFL = 0.725 # div damp
    #CFL = 0.7125 # org
    CFL = 0.6
    #CFL = 0.5
    #CFL = 0.2
    #CFL = 0.1
    cs = 340.0 # speed of sound (m/s)
    dt_raw = CFL * p_initial.dx * p_initial.ns / cs
    Nint_steps = ceil(Int, p_initial.tout_int / dt_raw)
    dt = T(p_initial.tout_int / Nint_steps) # Ensure dt is FT
    N_steps = ceil(Int, p_initial.t_end / dt)
    #println("dt = $dt, N_steps = $N_steps, Nint_steps = $Nint_steps, dt_raw = $dt_raw")

    # Axis
    x = Vector{T}(undef, p_initial.ia)
    zf = Vector{T}(undef, p_initial.ka)
    dz = Vector{T}(undef, p_initial.ka)
    z = Vector{T}(undef, p_initial.ka)
    for i in 1:p_initial.ia
        x[i] = (i - p_initial.is + FT(0.5)) * p_initial.dx
    end
    zf[p_initial.ks] = p_initial.dz0
    dz[p_initial.ks] = p_initial.dz0
    z[p_initial.ks] = zf[p_initial.ks] * FT(0.5)
    for k in p_initial.ks+1:p_initial.ke
        dz[k] = min(dz[k-1] * p_initial.z_fact, (p_initial.H - zf[k-1]) / (p_initial.ke - k + 1))
        zf[k] = zf[k-1] + dz[k]
        z[k] = (zf[k-1] + zf[k]) * FT(0.5)
    end
    if zf[p_initial.ke] < p_initial.H
        error("z_fact is too small: $(p_initial.z_fact), $(zf[p.ke]) < $(p_initial.H)")
    end

    # Create a new Params instance with calculated values
    return Params{T}(; kwargs...,
        x=x, zf=zf, dz=dz, z=z,
        dt=dt, N_steps=N_steps, Nint_steps=Nint_steps) # `p...` expands existing fields
end

# ============================================================
# 2. Device helper
# ============================================================
#device(x) = CUDA.has_cuda_gpu() ? CuArray(x) : x
device(x) = x # Force CPU for now

# ============================================================
# 3. Halo exchange for periodic BC
# ============================================================
function exchange_halo!(A, p::Params)
    # x-direction periodic
    @views begin
        # Copy right interior cells to left halo
        A[:, 1:p.halo] .= A[:, (p.ie-p.halo+1):p.ie]
        # Copy left interior cells to right halo
        A[:, (p.ie+1):(p.ie+p.halo)] .= A[:, p.is:(p.is+p.halo-1)]
    end
end

# ============================================================
# 4. State with halos on C-grid
# ============================================================
mutable struct State{A,B,C}
    # A: 2D (z,x), B: 1D (x), C: 1D (z)
    rho::A       # Total density (including vapor and water contents) (kg/m^3)
    rho_u::A     # Zonal momentum density (kg/m^2/s) @ (k, i+1/2)
    rho_w::A     # Vertical momentum density (kg/m^2/s) @ (k+1/2, i)
    rho_theta::A # Potential temperature density (K kg/m^3)
    rho_qv::A    # Water vapor density (kg/m^3)
    rho_qc::A    # Cloud water density (kg/m^3)
    rho_qr::A    # Rain water density (kg/m^3)
    prec::B      # Accumulated precipitation (kg/m^2)
    mflux::B     # Momentum flux (kg/m/s^2)
    shflux::B    # Surface heat flux (W/m^2)
    lhflux::B    # Latent heat flux (W/m^2)
    theta_ref::C # Reference potential temperature (K)
    d_rho_theta_mp::A # Tendency of mass-weighted potential temperature by cloud microphysics (kg K/m^3/s)
    d_rho_qv_mp::A    # Tendency of mass of vapor by cloud microphysics (kg/m^3/s)
    d_rho_qc_mp::A    # Tendency of mass of cloud water by cloud microphysics (kg/m^3/s)
    d_rho_qr_mp::A    # Tendency of mass of rain water by cloud microphysics (kg/m^3/s)
    d_rho_qr_sed::A   # Tendency of mass of rain water by rain sedimentation (kg/m^3/s)
    d_rho_u_sf::B     # Tendency of horizontal momentume by surface flux (kg/m^2/s^2)
    d_rho_theta_sf::B # Tendency of mass-weighted potential temperature by surface flux (kg K/m^3/s)
    d_rho_qv_sf::B    # Tendency of mass of vapor by surface flux (kg/m^3/s)
    d_rho_theta_rd::A # Tendency of mass-weighted potential temperature by radiation (kg K/m^3/s)
end

function allocate_state(p::Params{T}) where {T}
    ia = p.ia
    ka = p.ka
    return State(
        device(zeros(T, ka, ia)), # rho
        device(zeros(T, ka, ia)), # rho_u
        device(zeros(T, ka, ia)), # rho_w
        device(zeros(T, ka, ia)), # rho_theta
        device(zeros(T, ka, ia)), # rho_qv
        device(zeros(T, ka, ia)), # rho_qc
        device(zeros(T, ka, ia)),  # rho_qr
        device(zeros(T, ia)), # prec (1D, x)
        device(zeros(T, ia)), # mflux (1D, x)
        device(zeros(T, ia)), # shflux (1D, x)
        device(zeros(T, ia)), # lhflux (1D, x)
        device(zeros(T, ka)), # theta_ref (1D, z)
        device(zeros(T, ka, ia)), # d_rho_theta_mp
        device(zeros(T, ka, ia)), # d_rho_qv_mp
        device(zeros(T, ka, ia)), # d_rho_qc_mp
        device(zeros(T, ka, ia)), # d_rho_qr_mp
        device(zeros(T, ka, ia)), # d_rho_qr_sed
        device(zeros(T, ia)), # d_rho_u_sf (1D, x)
        device(zeros(T, ia)), # d_rho_theta_sf (1D, x)
        device(zeros(T, ia)), # d_rho_qv_sf (1D, x)
        device(zeros(T, ka, ia)), # d_rho_theta_rd
    )
end

# ============================================================
# 5. Initialize interior and exchange halos
# ============================================================

# Helper function for saturation vapor pressure
@inline function saturation_vapor_pressure(T_k::T) where {T}
    T_c = T_k - T0
    tmp = B_SAT + T_c
    es = ES0 * exp(A_SAT * T_c / tmp)
    des_dT = es * A_SAT * (tmp - T_c) / tmp^2
    return es, des_dT
end

# Helper function for saturation specific humidity
@inline function saturation_specific_humidity(p::T, es::T) where {T}
    tmp = p - (T(1.0) - EPSvap) * es
    if tmp > EPS
        qsat = EPSvap * es / tmp
        dqsat_des = EPSvap * p / tmp^2
    else
        qsat = T(0.0)
        dqsat_des = T(1.0)
    end
    return qsat, dqsat_des
end

@inline function f2h(vf1, vf2, dz1, dz2)
    vh = vf1 * dz2 + vf2 * dz1 / (dz1 + dz2)
    return vh
end

function init_state!(s::State, p::Params{T}, rngs::Vector{<:AbstractRNG}) where {T}
    @unpack is, ie, ka, ks, ke, Lx, H, dx, z = p
    @unpack u0, theta0, bndlayer_depth, z_tropopose, dtheta_dz_trop, dtheta_dz_stra, theta_pert, rel_hum_surf, rel_hum_decay = p

    # Temporary arrays for initialization
    theta_init = similar(s.rho_theta)
    qv_init = similar(s.rho_qv)

    # --- 1. Initialize Potential Temperature (theta) ---
    for k in ks:ke
        if z[k] < bndlayer_depth # Planetary boundary layer
            s.theta_ref[k] = theta0
        elseif z[k] < z_tropopose # Troposphere
            s.theta_ref[k] = theta0 + dtheta_dz_trop * (z[k] - bndlayer_depth)
        else # Stratosphere
            s.theta_ref[k] = theta0 + dtheta_dz_trop * (z_tropopose - bndlayer_depth) + dtheta_dz_stra * (z[k] - z_tropopose)
        end
    end
    xc, zc, xr, zr = Lx / 2, 2e3, 2e3, 2e3
    Threads.@threads for i in is:ie
        for k in ks:ke
            theta_init[k, i] = s.theta_ref[k]

            rng = rngs[Threads.threadid()]
            theta_init[k, i] += theta_pert * (rand(rng, T) * T(2.0) - T(1.0)) # Perturbation

            #=
            r = sqrt(((s.x[i] - xc) / xr)^2 + ((s.z[k] - zc) / zr)^2)
            if r < T(1.0) && i >= is && i <= ie # Apply perturbation only in interior
                #theta_init[k, i] += theta_pert * exp(-r^2)
                theta_init[k, i] += theta_pert * cos(pi * r * T(0.5))^2
            end
            =#
        end
    end

    # --- 2. Calculate Hydrostatic Balance initializing qv ---
    # Assume a simple relative humidity profile decreasing with height
    RH_surf = rel_hum_surf / 100.0 # Convert to fraction
    qv_ks = T(0.0) # first guess at ks
    Threads.@threads for i in is:ie
        # Assume a simple relative humidity profile decreasing with height
        Pi_k = T(1.0)
        local p_k, p_km, T_k
        for k in ks:ke
            # first guess
            if k == ks
                theta = (theta0 + theta_init[ks, i]) * T(0.5)
                qv_init[k, i] = qv_ks
                zm = T(0.0)
            else
                theta = (theta_init[k-1, i] + theta_init[k, i]) * T(0.5)
                qv_init[k, i] = qv_init[k-1, i]
                zm = z[k-1]
            end
            qd = T(1.0) - qv_init[k, i]
            r = Rd * qd + Rv * qv_init[k, i]
            cp = Cpd * qd + Cpv * qv_init[k, i]
            Pi_k = Pi_k - GRAV * (z[k] - zm) / (cp * theta)
            p_k = P0 * Pi_k^(cp / r)
            T_k = s.theta_ref[k] * Pi_k
            s.rho[k, i] = p_k / (r * T_k)
            for iter in 1:10
                # Calculate saturation vapor pressure and specific humidity
                es, _ = saturation_vapor_pressure(T_k)
                qsat, _ = saturation_specific_humidity(p_k, es)
                qv_new = max(T(0.0), RH_surf * exp(-z[k] / rel_hum_decay) * qsat)
                qv_init[k, i] += T(0.8) * (qv_new - qv_init[k, i])
                qd = T(1.0) - qv_init[k, i]
                r = Rd * qd + Rv * qv_init[k, i]
                cp = Cpd * qd + Cpv * qv_init[k, i]
                cv = cp - r
                Pi_k = (s.rho[k, i] * r * theta_init[k, i] / P0)^(r / cv)
                p_k = P0 * Pi_k^(cp / r)
                T_k = s.theta_ref[k] * Pi_k
                if k > ks
                    l = p_k - p_km + GRAV * (z[k] - z[k-1]) * T(0.5) * (s.rho[k-1, i] + s.rho[k, i])
                    dldr = cp / cv * p_k / s.rho[k, i] + GRAV * (z[k] - z[k-1]) * T(0.5)
                    s.rho[k, i] -= l / dldr
                end
            end
            p_km = p_k
        end
    end

    # --- 3. Set initial states ---
    fill!(s.rho_w, T(0.0))
    fill!(s.rho_qc, T(0.0))
    fill!(s.rho_qr, T(0.0))
    # rho_theta and rho_qv
    Threads.@threads for i in is:ie
        @turbo for k in ks:ke
            s.rho_u[k, i] = s.rho[k, i] * u0
            s.rho_theta[k, i] = s.rho[k, i] * theta_init[k, i]
            s.rho_qv[k, i] = s.rho[k, i] * qv_init[k, i]
        end
    end

    return s
end

function invert_T_for_thetae(qt::T, theta_e::T, p_k::T) where {T}
    T_low, T_high = T(250.0), T(350.0)
    local qsat
    for iter in 1:30
        t_mid = (T_low + T_high) / 2
        es, _ = saturation_vapor_pressure(t_mid)
        qsat, _ = saturation_specific_humidity(p_k, es)
        p_d = p_k - es
        qd = T(1.0) - qt
        cp = Cpd * qd + Cpl * qt
        theta_mid = t_mid * (p_d / P0)^(-Rd / cp) * exp(Lv * qsat / (cp * t_mid))
        theta_mid > theta_e ? (T_high = t_mid) : (T_low = t_mid)
    end
    return (T_low + T_high) / 2, qsat
end

# Bryan and Fritsch (2002) Moist experiment
function init_state_BF2002!(s::State, p::Params{T}) where {T}
    @unpack is, ie, ka, ks, ke, Lx, H, dx, x, z = p

    qt = T(0.02)    # qv + qc
    theta_e = T(320.0) # moist equivalent potential temperature
    qd = T(1.0) - qt # ratio of dry air

    p_col = similar(s.rho, ke)
    Pi_col = similar(p_col)

    # --- Calculate first guess ---
    # at the surface
    t_s, qv = invert_T_for_thetae(qt, theta_e, P0)
    theta = t_s
    r = Rd * qd + Rv * qv
    cp = Cpd * qd + Cpv * qv + Cpl * (qt - qv)
    # at ks
    Pi_col[ks] = T(1.0) - GRAV * z[ks] / (cp * theta)
    p_col[ks] = P0 * Pi_col[ks]^(cp / r)
    t_k, qv = invert_T_for_thetae(qt, theta_e, p_col[ks])
    theta = t_k / Pi_col[ks]
    for k in ks+1:ke
        r = Rd * qd + Rv * qv
        cp = Cpd * qd + Cpv * qv + Cpl * (qt - qv)
        Pi_col[k] = Pi_col[k-1] - GRAV * (z[k] - z[k-1]) / (cp * theta)
        p_col[k] = P0 * Pi_col[k]^(cp / r)
        t_k, qv = invert_T_for_thetae(qt, theta_e, p_col[k])
        theta = t_k / Pi_col[k]
    end
    max_iter = 100
    omega = T(0.7)
    for iter in 1:max_iter
        max_diff = T(0.0)
        # at k
        t_k, qv = invert_T_for_thetae(qt, theta_e, p_col[ks])
        r = Rd * qd + Rv * qv
        rho_s = P0 / (r * t_s) # at surface
        rho_k = p_col[ks] / (r * t_k)
        p_new = P0 - GRAV * (rho_s + rho_k) * (z[k] - z[k-1]) * T(0.25)
        p_diff = omega * (p_new - p_col[ks])
        p_col[ks] += p_diff
        max_diff = max(max_diff, abs(p_diff / p_new))
        for k in ks+1:ke
            t_km, qv = invert_T_for_thetae(qt, theta_e, p_col[k-1])
            r = Rd * qd + Rv * qv
            rho_km = p_col[k-1] / (r * t_km)
            t_k, qv = invert_T_for_thetae(qt, theta_e, p_col[k])
            r = Rd * qd + Rv * qv
            rho_k = p_col[k] / (r * t_k)
            p_new = p_col[k-1] - GRAV * (rho_km + rho_k) * T(0.5) * (z[k] - z[k-1])
            p_diff = omega * (p_new - p_col[k])
            p_col[k] += p_diff
            max_diff = max(max_diff, abs(p_diff / p_new))
        end
        if max_diff < 1e-10
            #println("$iter $max_diff")
            break
        end
        if iter == max_iter
            @warn "Hydrostatic balance did not converge: max p difference = $max_diff %"
        end
    end
    #println(p_col)
    #println(theta_col)

    xc, zc, xr, zr = Lx / 2, 2e3, 2e3, 2e3
    Threads.@threads for i in is:ie
        for k in ks:ke

            local t_k, qv, theta, r, cp
            t_k, qv = invert_T_for_thetae(qt, theta_e, p_col[k])
            r = Rd * qd + Rv * qv
            cp = Cpd * qd + Cpv * qv + Cpl * (qt - qv)
            theta = t_k * (P0 / p_col[k])^(r / cp)

            dist = sqrt(((x[i] - xc) / xr)^2 + ((z[k] - zc) / zr)^2)
            if dist < T(1.0) && i >= is && i <= ie # Apply perturbation only in interior
                tdiff = p.theta_pert * cos(pi * dist * T(0.5))^2 / T(300.0)
                theta = theta * (tdiff + T(1.0))
                #theta /= (T(1.0) - qt + qv / EPSvap)
            end
            es, _ = saturation_vapor_pressure(t_k)
            qsat, _ = saturation_specific_humidity(p_col[k], T(es))
            qv = min(qt, qsat)
            qc = qt - qv
            r = Rd * qd + Rv * qv

            s.rho[k, i] = p_col[k] / (r * t_k)
            s.rho_u[k, i] = s.rho[k, i] * p.u0
            s.rho_w[k, i] = T(0.0)
            s.rho_theta[k, i] = s.rho[k, i] * theta
            s.rho_qv[k, i] = s.rho[k, i] * qv
            s.rho_qc[k, i] = s.rho[k, i] * qc
            s.rho_qr[k, i] = T(0.0)
        end
    end

    return s
end

# --- Helper Function: Cloud Microphysics Tendencies ---
function cloud_microphysics!(d_rho_qv_phys, d_rho_qc_phys, d_rho_qr_phys, d_rho_theta_phys,
    rho, rho_theta, rho_qv, rho_qc, rho_qr, p::Params{T}) where {T}

    # Loop over interior points
    Threads.@threads for i in p.is:p.ie
        for k in p.ks:p.ke

            # Initialize tendencies to zero
            d_rho_qv_phys[k, i] = T(0.0)
            d_rho_qc_phys[k, i] = T(0.0)
            d_rho_qr_phys[k, i] = T(0.0)
            d_rho_theta_phys[k, i] = T(0.0)

            # --- Calculate thermodynamic variables ---
            rho_k = rho[k, i]
            # Avoid division by zero
            rho_inv = T(1.0) / rho_k

            theta_k = rho_theta[k, i] * rho_inv
            qv_k = rho_qv[k, i] * rho_inv
            qc_k = rho_qc[k, i] * rho_inv
            qr_k = rho_qr[k, i] * rho_inv

            qd = T(1.0) - qv_k - qc_k - qr_k
            r = Rd * qd + Rv * qv_k
            cp = Cpd * qd + Cpv * qv_k + Cpl * (qc_k + qr_k)
            cv = cp - r

            # Exner function Pi = (p/p0)^(Rd/cp)
            Pi_k = (r * rho_theta[k, i] / P0)^(r / cv)
            # Temperature T = theta * Pi
            T_k = theta_k * Pi_k
            # Pressure p = p0 * Pi^(cp/Rd)
            p_k = P0 * Pi_k^(cp / r)

            latent_heat_eff = Lv / (cp * Pi_k) # Effective heating term for d(theta)/dt

            # --- Saturation Adjustment (Condensation/Evaporation) ---
            max_iter = 3
            qv_init = qv_k
            local cond_evap, qsat_k
            for iter in 1:max_iter
                es_k, des_dT = saturation_vapor_pressure(T_k)
                qsat_k, dqsat_des = saturation_specific_humidity(p_k, FT(es_k))
                dqsat_dT = dqsat_des * des_dT
                cond_evap = (qv_k - qsat_k) / (T(1.0) + (Lv / cp) * dqsat_dT)
                # if qv_k > qsat_k then condensation, else evaporation
                cond_evap = min(cond_evap, qv_k) # Limit condensation by available vapor
                cond_evap = max(cond_evap, -qc_k) # Limit evaporation by available cloud water (This can be act as negative fixer.)
                qv_k -= cond_evap
                qc_k += cond_evap
                theta_k += cond_evap * latent_heat_eff

                r = Rd * qd + Rv * qv_k
                cp = Cpd * qd + Cpv * qv_k + Cpl * (qc_k + qr_k)
                cv = cp - r
                Pi_k = (r * rho[k, i] * theta_k / P0)^(r / cv)
                T_k = theta_k * Pi_k
                p_k = P0 * Pi_k^(cp / r)
                latent_heat_eff = Lv / (cp * Pi_k)
            end
            cond_evap = qv_init - qv_k
            cond_evap_rate = cond_evap * rho_k / p.dt
            d_rho_qv_phys[k, i] -= cond_evap_rate
            d_rho_qc_phys[k, i] += cond_evap_rate
            d_rho_theta_phys[k, i] += cond_evap_rate * latent_heat_eff

            # --- Autoconversion (Cloud Water -> Rain Water) ---
            if p.do_auto
                rate_auto = K_auto * max(T(0.0), qc_k - Qc_threshold)
                rate_auto = min(rate_auto, qc_k / p.dt)
                qc_k -= rate_auto * p.dt
                qr_k += rate_auto * p.dt
                rate_auto *= rho_k
                d_rho_qc_phys[k, i] -= rate_auto
                d_rho_qr_phys[k, i] += rate_auto
            end

            # --- Accretion (Rain collecting Cloud Water) ---
            if p.do_accr
                rate_accr = T(0.0)
                if qr_k > T(1e-12) && qc_k > T(1e-12)
                    # Simplified rate, often depends on qr^b or similar
                    rate_accr = K_accr * qc_k * qr_k # Simple rate proportional to both qc and qr
                    # Limit by available cloud water
                    rate_accr = min(rate_accr, qc_k / p.dt)
                end
                qc_k -= rate_accr * p.dt
                qr_k += rate_accr * p.dt
                rate_accr *= rho_k
                d_rho_qc_phys[k, i] -= rate_accr
                d_rho_qr_phys[k, i] += rate_accr
            end

            # --- Rain Evaporation ---
            if p.do_revap
                rate_evap = T(0.0)
                if qr_k > T(1e-12) && qv_k < qsat_k
                    # Rate depends on subsaturation and rain amount
                    # Simplified form:
                    rate_evap = K_evap * (qsat_k - qv_k) * sqrt(rho_k * qr_k) # Example dependency
                    # Limit by available rain water
                    rate_evap = min(rate_evap, qr_k) / p.dt
                end
                rate_evap *= rho_k
                d_rho_qr_phys[k, i] -= rate_evap
                d_rho_qv_phys[k, i] += rate_evap
                # Cooling due to rain evaporation
                r = Rd * qd + Rv * qv_k
                cp = Cpd * qd + Cpv * qv_k + Cpl * (qc_k + qr_k)
                cv = cp - r
                Pi_k = (r * rho[k, i] * theta_k / P0)^(r / cv)
                latent_heat_eff = Lv / (cp * Pi_k)
                d_rho_theta_phys[k, i] -= rate_evap * latent_heat_eff
            end
        end
    end
end

# --- Helper Function: Rain Sedimentation Tendency ---
function rain_sedimentation!(d_rho_qr_sed, d_prec, rho_qr, rho, p::Params{T}) where {T}

    # Calculate terminal velocity Vt at cell centers
    Vt = similar(rho_qr)
    Threads.@threads for i in p.is:p.ie
        @turbo for k in p.ks:p.ke
            # Avoid issues with zero or negative values
            rho_qr_eff = max(T(0.0), rho_qr[k, i])
            # Terminal velocity (downward is positive in this context)
            # Vt = a * (rho * qr)^b - assuming rho_air ~ rho
            Vt[k, i] = ifelse(rho_qr_eff > T(1e-12),
                A_term * (rho_qr_eff / rho[k, i])^B_term, # Simplified: uses rho*qr directly
                T(0.0))
        end
    end

    # Calculate flux divergence (centered difference)
    # d(rho*qr)/dt = - d(Flux_qr)/dz = - d(rho*qr*Vt)/dz
    # 1st-order upwind scheme
    Threads.@threads for i in p.is:p.ie
        @turbo for k in p.ks:p.ke
            # Note: Vt is defined as positive for downward motion
            # Flux at face k+1/2 (upper face)
            flux_kp = (k == p.ke) ? T(0.0) : rho_qr[k+1, i] * Vt[k+1, i]

            # Flux at face k-1/2 (lower face)
            flux_km = rho_qr[k, i] * Vt[k, i]

            # Divergence (downward flux is positive)
            d_rho_qr_sed[k, i] = (flux_kp - flux_km) / p.dz[k]
        end
        d_prec[i] = rho_qr[p.ks, i] * Vt[p.ks, i] # Precipitation flux at the surface
    end
end

# --- Helper Function: Surface flux Tendency ---
function surface_fluxes!(d_rho_u_sf, d_rho_theta_sf, d_rho_qv_sf, mflux, shflux, lhflux, rho, rho_u, rho_theta, rho_qv, rho_qc, rho_qr, p::Params{T}) where {T}
    @unpack ks, is, ie = p
    @unpack Cd, Ch, Ce, temp_surf, uabs_min = p

    Threads.@threads for i in is:ie
        qv = rho_qv[ks, i] / rho[ks, i]
        qc = rho_qc[ks, i] / rho[ks, i]
        qr = rho_qr[ks, i] / rho[ks, i]
        qd = T(1.0) - qv - qc - qr
        r = Rd * qd + Rv * qv
        cp = Cpd * qd + Cpv * qv + Cpl * (qc + qr)
        cv = cp - r

        uabs = abs(rho_u[ks, i-1] + rho_u[ks, i]) * T(0.5) / rho[ks, i] # at cell center (i)
        uabs = max(uabs, uabs_min)

        # Surface variables
        theta_ks = rho_theta[ks, i] / rho[ks, i] # Potential temperature at ks
        pi_ks = (r * rho_theta[ks, i] / P0)^(r / cv) # Exner function at ks
        pi_surf = pi_ks + dz[ks] * T(0.5) * GRAV / (cp * theta_ks) # Exner function at surface
        p_surf = P0 * pi_surf^(cp / r) # Pressure at surface
        rho_surf = p_surf / (r * temp_surf) # Density at surface
        theta_surf = temp_surf / pi_surf # Potential temperature at surface


        # ---- rho_u ----
        mflux[i] = rho_surf * Cd * (rho_u[ks, i] / (rho[ks, i] + rho[ks, i+1]) * T(2.0))^2 # Momentum flux (kg/m/s2) at i=1/2
        d_rho_u_sf[i] = -mflux[i] / p.dz[ks]

        # ---- rho_theta ----
        sh = rho_surf * Ch * uabs * (theta_surf - theta_ks) * pi_surf
        d_rho_theta_sf[i] = sh / (pi_ks * p.dz[ks])
        shflux[i] = sh * cp # Sensible heat flux (W/m2)

        # ---- rho_qv ----
        es, = saturation_vapor_pressure(temp_surf)
        qsat_s, = saturation_specific_humidity(p_surf, es)
        lh = rho_surf * Ce * uabs * (qsat_s - qv)
        d_rho_qv_sf[i] = lh / p.dz[ks]
        lhflux[i] = lh * Lv # Latent heat flux (W/m2)
    end
end

# --- Helper Function: Radiative Cooling Tendency ---
function radiative_cooling!(d_rho_theta_rd, rho, rho_theta, rho_qv, rho_qc, rho_qr, p::Params{T}, rad_cooling::T) where {T}
    @unpack ks, ke, is, ie = p

    Threads.@threads for i in is:ie
        @turbo for k in p.ks:p.ke
            # Radiative cooling tendency (kg K/m3/s)
            # rad_cooling is in K/s
            qv = rho_qv[k, i] / rho[k, i]
            qc = rho_qc[k, i] / rho[k, i]
            qr = rho_qr[k, i] / rho[k, i]
            qd = T(1.0) - qv - qc - qr
            r = Rd * qd + Rv * qv
            cp = Cpd * qd + Cpv * qv + Cpl * (qc + qr)
            cv = cp - r
            pi = (r * rho_theta[k, i] / P0)^(r / cv) # Exner function
            d_rho_theta_rd[k, i] = -rad_cooling * rho[k, i] / pi
        end
    end
end

# --- Helper Function: Gray Radiation Tendency ---
function gray_radiation!(d_rho_theta_rd, rho, rho_theta, rho_qv, rho_qc, rho_qr, theta_ref, p::Params{T}) where {T}
    @unpack ks, ke, is, ie = p
    @unpack z_tropopose, k_abs_qv, k_abs_qc, k_abs_qr, surf_emissivity, tau_cooling, temp_surf = p

    LW_up = device(similar(rho, ke + 1, ie))
    LW_dn = device(similar(rho, ke + 1, ie))
    epsilon = device(similar(rho))
    tau = device(similar(rho))
    temp = device(similar(rho))
    pres = device(similar(rho))
    pi_exner = device(similar(rho))
    cp = device(similar(rho))

    fill!(d_rho_theta_rd, T(0.0))

    Threads.@threads for i in is:ie
        @turbo for k in ks:ke
            qv = rho_qv[k, i] / rho[k, i]
            qc = rho_qc[k, i] / rho[k, i]
            qr = rho_qr[k, i] / rho[k, i]
            qd = T(1.0) - qv - qc - qr
            r = Rd * qd + Rv * qv
            cp[k, i] = Cpd * qd + Cpv * qv + Cpl * (qc + qr)
            cv = cp[k, i] - r
            pi_exner[k, i] = (r * rho_theta[k, i] / P0)^(r / cv) # Exner function
            temp[k, i] = rho_theta[k, i] / rho[k, i] * pi_exner[k, i]
            pres[k, i] = P0 * pi_exner[k, i]^(cp[k, i] / r)

            # Optimal thickness and emissivity/transmissivity
            delta_opt_qv = k_abs_qv * rho_qv[k, i] * p.dz[k]
            delta_opt_qc = k_abs_qc * rho_qc[k, i] * p.dz[k]
            delta_opt_qr = k_abs_qr * rho_qr[k, i] * p.dz[k]
            delta_opt = delta_opt_qv + delta_opt_qc + delta_opt_qr
            epsilon[k, i] = T(1.0) - exp(-delta_opt) # Emissivity
            tau[k, i] = T(1.0) - epsilon[k, i] # Transmissivity
        end

        # --- Calculate LW fluxes ---

        # Upward flux
        LW_up[ks, i] = surf_emissivity * SIGMA * temp_surf^4
        for k in ks:ke
            LW_up[k+1, i] = LW_up[k, i] * tau[k, i] + epsilon[k, i] * SIGMA * temp[k, i]^4
        end

        # Downward flux
        LW_dn[ke+1, i] = T(0.0) # Assume zero downward flux from space
        for k in ke:-1:ks
            LW_dn[k, i] = LW_dn[k+1, i] * tau[k, i] + epsilon[k, i] * SIGMA * temp[k, i]^4
        end

        # --- Calculate radiative heating rate ---
        dz_trans = T(2.0e3) # 2 km
        for k in ks:ke
            # Heating rate by long wave flux
            divF = ((LW_up[k+1, i] - LW_dn[k+1, i]) - (LW_up[k, i] - LW_dn[k, i])) / p.dz[k]
            hr = - divF / (cp[k, i] * pi_exner[k, i])
            # Newtonian cooling in stratosphere
            alpha = T(0.5) * (T(1.0) + tanh((z[k] - (z_tropopose + dz_trans * T(0.5)))/ dz_trans)) / tau_cooling # (1/s)
            cr = - (rho_theta[k, i]  - rho[k, i] * theta_ref[k]) * alpha
            # Total tendency
            d_rho_theta_rd[k, i] = hr + cr
        end
    end
end


# --- Helper Function: Koren Limiter ---
@inline function koren_limiter(r::T) where {T}
    # Koren (1993) limiter
    return max(T(0.0), min(T(2.0) * r, min((T(1.0) + T(2.0) * r) / T(3.0), T(2.0))))
end

# --- Helper Function: Calculate Limited Flux in X-direction ---
@inline function compute_limited_flux_x(scalar_rho, rho, rho_u, k::Int, i::Int, p::Params{T}) where {T}
    # scalar_rho: conserved variable (rho * s) at cell centers
    # rho: density at cell centers
    # rho_u: momentum density at x-faces (rho * u) -> rho_u[k, i] is at face (k, i+1/2)

    @unpack is, ie, dx = p

    # Face velocity (momentum density)
    rhou_face = rho_u[k, i] # at face i+1/2

    # Get scalar values (s = scalar_rho / rho) in neighboring cells
    # Need cells i-1, i, i+1, i+2 for 3rd order upwind
    # Ensure indices are within bounds (handle halo/boundaries)
    im1 = i - 1
    ip1 = i + 1
    ip2 = i + 2

    s_i = scalar_rho[k, i] / rho[k, i]
    s_ip1 = scalar_rho[k, ip1] / rho[k, ip1]

    if rhou_face > EPS # Wind from left (use cells i-1, i, i+1)
        s_im1 = scalar_rho[k, im1] / rho[k, im1]

        r_num = s_ip1 - s_i
        r_den = s_i - s_im1
        r = ifelse(abs(r_den) < EPS, T(1.0), r_num / r_den)
        phi = koren_limiter(r)
        s_face = s_i + T(0.5) * phi * r_den
    elseif rhou_face < -EPS # Wind from right (use cells i+2, i+1, i)
        s_ip2 = scalar_rho[k, ip2] / rho[k, ip2]

        r_num = s_i - s_ip1
        r_den = s_ip1 - s_ip2
        r = ifelse(abs(r_den) < EPS, T(1.0), r_num / r_den)
        phi = koren_limiter(r)
        s_face = s_ip1 + T(0.5) * phi * r_den
    else # Zero velocity
        s_face = T(0.5) * (s_i + s_ip1) # 2nd order centered
    end

    # Flux = face_value * momentum_density
    return s_face * rhou_face
end

# --- Helper Function: Calculate Limited Flux in Z-direction ---
@inline function compute_limited_flux_z(scalar_rho, rho, rho_w, k::Int, i::Int, p::Params{T}) where {T}
    # scalar_rho: conserved variable (rho * s) at cell centers
    # rho: density at cell centers
    # rho_w: momentum density at z-faces (rho * w) -> rho_w[k, i] is at face (k+1/2, i)

    @unpack ks, ke, dz, dt = p

    # Face velocity (momentum density)
    rhow_face = rho_w[k, i] # at face k+1/2

    # Get scalar values (s = scalar_rho / rho) in neighboring cells
    # Need cells k-1, k, k+1, k+2 for 3rd order upwind
    km1 = k - 1
    kp1 = k + 1
    kp2 = k + 2

    s_k = scalar_rho[k, i] / rho[k, i]
    s_kp1 = scalar_rho[kp1, i] / rho[kp1, i]

    if rhow_face > EPS # Wind from below (use cells k-1, k, k+1)
        if k == ks
            s_face = f2h(s_k, s_kp1, dz[k], dz[kp1]) # 2nd order near boundary
            s_face = min(s_face, scalar_rho[k] / (rhow_face * dt)) # limiter to avoid negative value
        else
            s_km1 = scalar_rho[km1, i] / rho[km1, i]

            r_num = s_kp1 - s_k
            r_den = s_k - s_km1
            r = ifelse(abs(r_den) < EPS, T(1.0), r_num / r_den)
            phi = koren_limiter(r)
            s_face = s_k + T(0.5) * phi * r_den
        end
    elseif rhow_face < -EPS # Wind from above (use cells k+2, k+1, k)
        if k == ke - 1
            s_face = f2h(s_k, s_kp1, dz[k], dz[k+1]) # 2nd order near boundary
            s_face = min(s_face, scalar_rho[k+1] / (-rhow_face * dt)) # limiter to avoid negative value
        else
            s_kp2 = scalar_rho[kp2, i] / rho[kp2, i]

            r_num = s_k - s_kp1
            r_den = s_kp1 - s_kp2
            r = ifelse(abs(r_den) < EPS, T(1.0), r_num / r_den)
            phi = koren_limiter(r)
            s_face = s_kp1 - T(0.5) * phi * r_den
        end
    else # Zero velocity
        s_face = f2h(s_k, s_kp1, dz[k], dz[k+1]) # 2nd order centered
    end

    return s_face * rhow_face
end

# --- Helper Function: Tridiagonal Solver (Thomas Algorithm) ---
# Solves A*x = d where A is tridiagonal with sub-diagonal a, diagonal b, super-diagonal c.
# a, b, c, d are vectors of length n.
# a[1] and c[n] are not used.
# Modifies c and d in place. Returns the solution d.
function solve_tridiagonal!(a::V, b::V, c::V, d::V) where {T,V<:AbstractVector{T}}
    n = length(b)
    local temp # Ensure temp is local to avoid race conditions if parallelized later
    # Forward elimination
    c_1 = c[1] / b[1]
    d_1 = d[1] / b[1]
    c[1] = c_1 # Store modified c[1]
    d[1] = d_1 # Store modified d[1]
    for i in 2:n
        temp = b[i] - a[i] * c[i-1] # Use stored modified c[i-1]
        c_i = c[i] / temp
        d_i = (d[i] - a[i] * d[i-1]) / temp # Use stored modified d[i-1]
        c[i] = c_i # Store modified c[i]
        d[i] = d_i # Store modified d[i]
    end
    # Back substitution
    for i in (n-1):-1:1
        d[i] = d[i] - c[i] * d[i+1]
    end
end

# --- Implicit Calculation for Vertical Sound Waves ---
function implicit_correction!(rho_w, drho_w, theta, rt2pres, dt::T, p::Params{T}) where {T}
    @unpack Nz, is, ie, ks, ke, z, dz, beta_offcenter = p

    # --- Build and solve the tridiagonal system for each vertical column i ---
    # The system solves for the updated rho_w^{n+1} implicitly.
    # A*rho_w^{n+1} = D


    theta_h = similar(theta)
    for k in ks:ke-1
        theta_h[k] = f2h(theta[k], theta[k+1], dz[k], dz[k+1]) # Potential temperature at face k+1/2
    end

    fact_tp = T(0.5) * (T(1.0) + beta_offcenter)
    fact_tm = T(0.5) * (T(1.0) - beta_offcenter)
    dt_tp2 = (dt * fact_tp)^2

    a = similar(rho_w, Nz - 1) # sub-diagonal (indices ks+1 to ke)
    b = similar(rho_w, Nz - 1) # diagonal (indices ks to ke-1)
    c = similar(rho_w, Nz - 1) # super-diagonal (indices ks to ke-2)
    d = similar(rho_w, Nz - 1) # right-hand side (indices ks to ke-1), output
    # --- Construct the tridiagonal system ---
    # Loop over vertical faces k (representing rho_w[k,i] at face k+1/2)
    for k in ks:ke-1 # rho_w lives on faces ks+1/2 to ke-1/2
        # Assign to solver arrays (adjusting indices)
        b[k-ks+1] = dt_tp2 * (rt2pres[k] / dz[k] + rt2pres[k+1] / dz[k+1]) * theta_h[k] / (z[k+1] - z[k])
        if k > ks
            a[k-ks+1] = -dt_tp2 * (rt2pres[k] * theta_h[k-1] / ((z[k+1] - z[k]) * dz[k]) - GRAV / (dz[k] + dz[k+1]))
        end # a[ks] is unused
        if k < ke - 1
            c[k-ks+1] = -dt_tp2 * (rt2pres[k+1] * theta_h[k+1] / ((z[k+1] - z[k]) * dz[k+1]) + GRAV / (dz[k] + dz[k+1]))
        end # c[ke-1] is unused

        d[k-ks+1] = rho_w[k] + drho_w[k]
    end

    # Apply off-center beta correction
    fact = fact_tm / fact_tp
    for k in ks:ke-1
        if k > ks
            d[k-ks+1] -= a[k-ks+1] * fact * rho_w[k]
        end
        d[k-ks+1] -= b[k-ks+1] * fact * rho_w[k]
        b[k-ks+1] += T(1.0)
        if k < ke - 1
            d[k-ks+1] -= c[k-ks+1] * fact * rho_w[k+1]
        end
    end

    # Solve the system for column i (rho_w[ks] to rho_w[ke-1])
    solve_tridiagonal!(a, b, c, d)

    # Update rho_w
    for k in ks:ke-1
        rho_w[k] = d[k-ks+1]
    end
end

# --- Main integration step ---
# Forward-Backward scheme
# 1. Compute tendencies (RHS) for slow modes
# 2. Small time step for fast modes
# 2.1 Update rho_u with explicit scheme
# 2.2 Update rho, rho_w, and rho_theta with horizontally explicit and vertically implicit scheme
# 3. Update rho_qv, rho_qc, rho_qr with momentum fluxes that are consistent with those used for updating rho
function compute_step!(d_rho_phys, d_rho_u_phys, d_rho_theta_phys, d_rho_qv_phys, d_rho_qc_phys, d_rho_qr_phys,
    s::State, s0::State, p::Params{T}, dt::T, ns::Int) where {T}

    @unpack dx, nu, kappa, gamma_d, beta_offcenter, zf_sponge = p

    # --- Precompute diagnostic variables ---
    pres = similar(s.rho)
    theta = similar(s.rho)
    rt2pres = similar(s.rho) # conversion factor from rho_theta perturbation to pressure perturbation
    Threads.@threads for i in p.is:p.ie+1
        @turbo for k in p.ks:p.ke
        #for k in p.ks:p.ke
            qv = s.rho_qv[k, i] / s.rho[k, i]
            qc = s.rho_qc[k, i] / s.rho[k, i]
            qr = s.rho_qr[k, i] / s.rho[k, i]
            qd = T(1.0) - qv - qc - qr
            r = Rd * qd + Rv * qv
            cp = Cpd * qd + Cpv * qv + Cpl * (qc + qr)
            cv = cp - r
            pres[k, i] = P0 * (r * s.rho_theta[k, i] / P0)^(cp / cv)
            theta[k, i] = s.rho_theta[k, i] / s.rho[k, i]
            rt2pres[k, i] = cp / cv * pres[k, i] / s.rho_theta[k, i] # cp / cv * R * exner
        end
    end
    exchange_halo!(pres, p)
    exchange_halo!(theta, p)
    exchange_halo!(rt2pres, p)

    exchange_halo!(s.rho, p)
    exchange_halo!(s.rho_u, p)
    exchange_halo!(s.rho_w, p)

    # --- Compute tendencies by slow modes (RHS) ---
    R_rho = similar(s.rho)
    R_rho_u = similar(s.rho_u)
    R_rho_w = similar(s.rho_w)
    R_rho_theta = similar(s.rho_theta)

    # --- R_rho (cell center) ---
    # d/dt(rho)
    Threads.@threads for i in p.is:p.ie
        for k in p.ks:p.ke
            flux_kp = k == p.ke ? T(0.0) : s.rho_w[k, i]
            flux_km = k == p.ks ? T(0.0) : s.rho_w[k-1, i]
            advection = -((s.rho_u[k, i] - s.rho_u[k, i-1]) / dx + (flux_kp - flux_km) / p.dz[k])
            R_rho[k, i] = advection + d_rho_phys[k, i]
        end
    end


    # --- R_rho_u (x-face) ---
    # d/dt(rho*u)
    u_face = similar(s.rho_u)
    Threads.@threads for i in p.is:p.ie
        @turbo for k in p.ks:p.ke
            u_face[k, i] = s.rho_u[k, i] * T(2.0) / (s.rho[k, i] + s.rho[k, i+1]) # u at (k,i+1/2)
        end
    end
    exchange_halo!(u_face, p)
    Threads.@threads for i in p.is:p.ie
        for k in p.ks:p.ke
            # Advection term: div(rho * u * vec(u)) at x-face (k, i+1/2)

            # d/dx(rho*u*u)
            # Flux at center (k, i+1):
            rho_u_ip = T(0.5) * (s.rho_u[k, i] + s.rho_u[k, i+1]) # rho_u at (k,i+1)
            if rho_u_ip > EPS # use i-1/2, i+1/2, i+2/3
                flux_uu_x_ip = rho_u_ip * (-u_face[k, i-1] + T(5.0) * u_face[k, i] + T(2.0) * u_face[k, i+1]) / T(6.0)
            elseif rho_u_ip < -EPS # use i+5/2, i+3/2, i+1/2
                flux_uu_x_ip = rho_u_ip * (-u_face[k, i+2] + T(5.0) * u_face[k, i+1] + T(2.0) * u_face[k, i]) / T(6.0)
            else
                #flux_uu_x_ip = rho_u_ip^2 / s.rho[k, i+1]
                flux_uu_x_ip = rho_u_ip * (u_face[k, i] + u_face[k, i+1]) * T(0.5)
            end
            # Flux at center (k, i):
            rho_u_im = T(0.5) * (s.rho_u[k, i-1] + s.rho_u[k, i]) # rho_u at (k,i)
            if rho_u_im > EPS # use i-3/2, i-1/2, i+1/2
                flux_uu_x_im = rho_u_im * (-u_face[k, i-2] + T(5.0) * u_face[k, i-1] + T(2.0) * u_face[k, i]) / T(6.0)
            elseif rho_u_im < -EPS # use i+3/2, i+1/2, i-1/2
                flux_uu_x_im = rho_u_im * (-u_face[k, i+1] + T(5.0) * u_face[k, i] + T(2.0) * u_face[k, i-1]) / T(6.0)
            else
                #flux_uu_x_im = rho_u_im^2 / s.rho[k, i]
                flux_uu_x_im = rho_u_im * (u_face[k, i-1] + u_face[k, i]) * T(0.5)
            end

            druudx = (flux_uu_x_ip - flux_uu_x_im) / dx # d/dx(rho*u*u) at (k,i+1/2)

            # d/dz(rho*u*w) term approximation
            # Flux at corner (k+1/2, i+1/2):
            if k == p.ke
                flux_uw_z_kp = T(0.0)
            else
                rho_w_kp = T(0.5) * (s.rho_w[k, i] + s.rho_w[k, i+1]) # rho_w at (k+1/2,i+1/2)
                if rho_w_kp > EPS && k > p.ks # use k-1, k, k+1
                    flux_uw_z_kp = rho_w_kp * (-u_face[k-1, i] + T(5.0) * u_face[k, i] + T(2.0) * u_face[k+1, i]) / T(6.0)
                elseif rho_w_kp < -EPS && k < p.ke - 1 # use k+2, k+1, k
                    flux_uw_z_kp = rho_w_kp * (-u_face[k+2, i] + T(5.0) * u_face[k+1, i] + T(2.0) * u_face[k, i]) / T(6.0)
                else
                    flux_uw_z_kp = rho_w_kp * f2h(u_face[k, i], u_face[k+1, i], p.dz[k], p.dz[k+1])
                end
            end
            # Flux at corner (k-1/2 i+1/2):
            if k == p.ks
                flux_uw_z_km = T(0.0)
            else
                rho_w_km = T(0.5) * (s.rho_w[k-1, i] + s.rho_w[k-1, i+1]) # rho_w at (k-1/2,i+1/2)
                if rho_w_km > EPS && k > p.ks + 1 # use k-2, k-1, k
                    flux_uw_z_km = rho_w_km * (-u_face[k-2, i] + T(5.0) * u_face[k-1, i] + T(2.0) * u_face[k, i]) / T(6.0)
                elseif rho_w_km < -EPS && k < p.ke # use k+1, k, k-1
                    flux_uw_z_km = rho_w_km * (-u_face[k+1, i] + T(5.0) * u_face[k, i] + T(2.0) * u_face[k-1, i]) / T(6.0)
                else
                    flux_uw_z_km = rho_w_km * f2h(u_face[k-1, i], u_face[k, i], p.dz[k-1], p.dz[k])
                end
            end
            drwudz = (flux_uw_z_kp - flux_uw_z_km) / p.dz[k] # d/dz(rho*w*u) at (k,i+1/2)

            advection = -druudx - drwudz

            # Viscous term: nu * div(rho * grad(u)) at x-face (k, i+1/2)

            # d/dx(rho * du/dx)
            u_ip = (u_face[k, i+1] - u_face[k, i]) / dx # du/dx at (k,i+1)
            u_im = (u_face[k, i] - u_face[k, i-1]) / dx # du/dx at (k,i)
            u_xx_term = (u_ip * s.rho[k, i+1] - u_im * s.rho[k, i]) / dx # d/dx(rho*du/dx) at (k,i+1/2)

            # d/dz(rho * du/dz)
            u_kp = k == p.ke ? T(0.0) : (u_face[k+1, i] - u_face[k, i]) / (p.z[k+1] - p.z[k]) # du/dz at (k+1/2,i+1/2)
            u_km = k == p.ks ? T(0.0) : (u_face[k, i] - u_face[k-1, i]) / (p.z[k] - p.z[k-1]) # du/dz at (k-1/2,i+1/2)
            rho_kp = k == p.ke ? T(0.0) : T(0.25) * (s.rho[k, i] + s.rho[k+1, i] + s.rho[k, i+1] + s.rho[k+1, i+1]) # rho at (k+1/2,i+1/2)
            rho_km = k == p.ks ? T(0.0) : T(0.25) * (s.rho[k-1, i] + s.rho[k, i] + s.rho[k-1, i+1] + s.rho[k, i+1]) # rho at (k-1/2,i+1/2)
            u_zz_term = (u_kp * rho_kp - u_km * rho_km) / p.dz[k] # d/dx(rho*du/dz) at (k,i+1/2)

            diffusion = nu * (u_xx_term + u_zz_term)

            # Pressure gradient in x-direction at x-face (k, i+1/2)
            dpdx = (pres[k, i+1] - pres[k, i]) / dx

            # Combine terms
            R_rho_u[k, i] = advection - dpdx + diffusion + d_rho_u_phys[k, i]
        end
    end
    #println("minmax R_rho_u: ", minimum(R_rho_u[p.ks:p.ke, p.is:p.ie]), " ", maximum(R_rho_u[p.ks:p.ke, p.is:p.ie]))

    # --- R_rho_w ---
    # d/dt(rho)
    w_face = similar(s.rho_w)
    Threads.@threads for i in p.is-2:p.ie+2
        for k in p.ks:p.ke-1
            w_face[k, i] = s.rho_w[k, i] / f2h(s.rho[k, i], s.rho[k+1, i], p.dz[k], p.dz[k+1]) # w at (k+1/2,i)
        end
    end
    Threads.@threads for i in p.is:p.ie
        for k in p.ks:p.ke-1
            # Advection term: div(rho * w * vec(u)) at z-face (k+1/2, i)

            # d/dx(rho*w*u)
            # Flux at corner (k+1/2,i+1/2):
            rho_u_ip = f2h(s.rho_u[k, i], s.rho_u[k+1, i], p.z[k], p.z[k+1]) # rho_u at (k+1/2,i+1/2)
            if rho_u_ip > EPS # use i-1, i, i+1
                flux_uw_x_ip = rho_u_ip * (-w_face[k, i-1] + T(5.0) * w_face[k, i] + T(2.0) * w_face[k, i+1]) / T(6.0)
            elseif rho_u_ip < -EPS # use i+2, i+1, i
                flux_uw_x_ip = rho_u_ip * (-w_face[k, i+2] + T(5.0) * w_face[k, i+1] + T(2.0) * w_face[k, i]) / T(6.0)
            else
                flux_uw_x_ip = rho_u_ip * (w_face[k, i] + w_face[k, i+1]) * T(0.5)
            end
            # Flux at corner (k+1/2,i-1/2):
            rho_u_im = f2h(s.rho_u[k, i-1], s.rho_u[k+1, i-1], p.dz[k], p.dz[k+1]) # rho_u at (k+1/2,i-1/2)
            if rho_u_im > EPS # use i-2, i-1, i
                flux_uw_x_im = rho_u_im * (-w_face[k, i-2] + T(5.0) * w_face[k, i-1] + T(2.0) * w_face[k, i]) / T(6.0)
            elseif rho_u_im < -EPS # use i+1, i, i-1
                flux_uw_x_im = rho_u_im * (-w_face[k, i+1] + T(5.0) * w_face[k, i] + T(2.0) * w_face[k, i-1]) / T(6.0)
            else
                flux_uw_x_im = rho_u_im * (w_face[k, i-1] + w_face[k, i]) * T(0.5)
            end
            drwudx = (flux_uw_x_ip - flux_uw_x_im) / dx # d/dx(rho*w*u) at (k+1/2, i)

            # d/dz(rho*w*w)
            # Flux at center (k+1,i):
            if k == p.ke - 1
                flux_ww_z_kp = T(0.0)
            else
                rho_w_kp = f2h(s.rho_w[k, i], s.rho_w[k+1, i], p.dz[k], p.dz[k+1]) # rho_w at (k+1,i)
                if rho_w_kp > EPS && k > p.ks # use k-1/2, k+1/2, k+3/2
                    flux_ww_z_kp = rho_w_kp * (-w_face[k-1, i] + T(5.0) * w_face[k, i] + T(2.0) * w_face[k+1, i]) / T(6.0)
                elseif rho_w_kp < -EPS && k < p.ke - 2 # use k+5/2, k+3/2, k+1/2
                    flux_ww_z_kp = rho_w_kp * (-w_face[k+2, i] + T(5.0) * w_face[k+1, i] + T(2.0) * w_face[k, i]) / T(6.0)
                else
                    flux_ww_z_kp = rho_w_kp * f2h(w_face[k, i], w_face[k+1, i], p.dz[k], p.dz[k+1])
                end
            end
            # Flux at center (k,i):
            if k == p.ks
                flux_ww_z_km = T(0.0)
            else
                rho_w_km = f2h(s.rho_w[k-1, i], s.rho_w[k, i], p.dz[k-1], p.dz[k]) # rho_w at (k,i)
                if rho_w_km > EPS && k > p.ks + 1 # use k-3/2, k-1/2, k+1/2
                    flux_ww_z_km = rho_w_km * (-w_face[k-2, i] + T(5.0) * w_face[k-1, i] + T(2.0) * w_face[k, i]) / T(6.0)
                elseif rho_w_km < -EPS && k < p.ke - 1 # use k+3/2, k+1/2, k-1/2
                    flux_ww_z_km = rho_w_km * (-w_face[k+1, i] + T(5.0) * w_face[k, i] + T(2.0) * w_face[k-1, i]) / T(6.0)
                else
                    flux_ww_z_km = rho_w_km * f2h(w_face[k-1, i],w_face[k, i], p.dz[k-1], p.dz[k])
                end
            end
            drwwdz = (flux_ww_z_kp - flux_ww_z_km) / (p.z[k+1] - p.z[k]) # d/dz(rho*w*w) at (k+1/2,i)
            advection = -drwudx - drwwdz

            # Viscous term: nu * div(rho * grad(w)) at z-face (k+1/2, i)

            # d/dx(rho * dw/dx)
            w_ip = (w_face[k, i+1] - w_face[k, i]) / dx # w at (k+1/2,i+1/2)
            w_im = (w_face[k, i] - w_face[k, i-1]) / dx # w at (k+1/2,i-1/2)
            rho_ip = T(0.25) * (s.rho[k, i] + s.rho[k+1, i] + s.rho[k, i+1] + s.rho[k+1, i+1]) # rho at (k+1/2,i+1/2)
            rho_im = T(0.25) * (s.rho[k, i-1] + s.rho[k+1, i-1] + s.rho[k, i] + s.rho[k+1, i]) # rho at (k+1/2,i-1/2)
            w_xx_term = (w_ip * rho_ip - w_im * rho_im) / dx # d/dx(rho*dw/dx) at (k+1/2,i)

            # d/dz(rho * dw/dz)
            w_kp = k == p.ke - 1 ? T(0.0) : (w_face[k+1, i] - w_face[k, i]) / p.dz[k+1] # w at (k+1,i)
            w_km = k == p.ks ? T(0.0) : (w_face[k, i] - w_face[k-1, i]) / p.dz[k] # w at (k,i)
            w_zz_term = (w_kp * s.rho[k+1, i] - w_km * s.rho[k, i]) / (p.z[k+1] - p.z[k]) # d/dz(rho*dw/dz) at (k+1/2,i)

            diffusion = nu * (w_xx_term + w_zz_term)


            # Sponge layer term (Rayleigh damping) near the top
            # Apply damping based on the height of the z-face
            sigma = T(0.0)
            if p.zf[k] > (T(1.0) - zf_sponge) * p.H
                sigma = (GRAV / T(30.0)) * ((p.zf[k] - (T(1.0) - zf_sponge) * p.H) / (zf_sponge * p.H))
            end
            sponge_term = -sigma * s.rho_w[k, i]

            # Pressure gradient in z-direction at z-face (k+1/2,i)
            dpdz = (pres[k+1, i] - pres[k, i]) / (p.z[k+1] - p.z[k])

            # Buoyancy term at z-face (k+1/2,i)
            buoyancy = -GRAV * (s.rho[k, i] + s.rho[k+1, i]) * T(0.5)

            # Combine terms
            R_rho_w[k, i] = advection - dpdz + buoyancy + diffusion + sponge_term
        end
    end
    #println("minmax R_rho_w: ", minimum(R_rho_w[p.ks:p.ke-1, p.is:p.ie]), " ", maximum(R_rho_w[p.ks:p.ke-1, p.is:p.ie]))

    # --- R_rho_theta ---
    # d/dt(rho*theta)
    Threads.@threads for i in p.is:p.ie
        for k in p.ks:p.ke

            # d/dx(rho*u*theta)
            # Flux at (k,i+1/2)
            if s.rho_u[k, i] > EPS # use i-1, i, i+1
                flux_ip = s.rho_u[k, i] * (-theta[k, i-1] + T(5.0) * theta[k, i] + T(2.0) * theta[k, i+1]) / T(6.0)
            elseif s.rho_u[k, i] < -EPS # use i+2, i+1, i
                flux_ip = s.rho_u[k, i] * (-theta[k, i+2] + T(5.0) * theta[k, i+1] + T(2.0) * theta[k, i]) / T(6.0)
            else
                flux_ip = s.rho_u[k, i] * (theta[k, i] + theta[k, i+1]) * T(0.5)
            end
            # Flux at (k,i-1/2)
            if s.rho_u[k, i-1] > EPS # use i-2, i-1, i
                flux_im = s.rho_u[k, i-1] * (-theta[k, i-2] + T(5.0) * theta[k, i-1] + T(2.0) * theta[k, i]) / T(6.0)
            elseif s.rho_u[k, i-1] < -EPS # use i+1, i, i-1
                flux_im = s.rho_u[k, i-1] * (-theta[k, i+1] + T(5.0) * theta[k, i] + T(2.0) * theta[k, i-1]) / T(6.0)
            else
                flux_im = s.rho_u[k, i-1] * (theta[k, i-1] + theta[k, i]) * T(0.5)
            end
            # d/dz(rho*w*theta)
            # Flux at (k+1/2,i)
            if k == p.ke
                flux_kp = T(0.0)
            else
                if s.rho_w[k, i] > EPS && k > p.ks # use k-1, k, k+1
                    flux_kp = s.rho_w[k, i] * (-theta[k-1, i] + T(5.0) * theta[k, i] + T(2.0) * theta[k+1, i]) / T(6.0)
                elseif s.rho_w[k, i] < -EPS && k < p.ke - 1 # use k+2, k+1, k
                    flux_kp = s.rho_w[k, i] * (-theta[k+2, i] + T(5.0) * theta[k+1, i] + T(2.0) * theta[k, i]) / T(6.0)
                else
                    flux_kp = s.rho_w[k, i] * f2h(theta[k, i], theta[k+1, i], p.dz[k], p.dz[k+1])
                end
            end
            # Flux at (k-1/2,i)
            if k == p.ks
                flux_km = T(0.0)
            else
                if s.rho_w[k-1, i] > EPS && k > p.ks + 1 # use k-2, k-1, k
                    flux_km = s.rho_w[k-1, i] * (-theta[k-2, i] + T(5.0) * theta[k-1, i] + T(2.0) * theta[k, i]) / T(6.0)
                elseif s.rho_w[k-1, i] < -EPS && k < p.ke # use k+1, k, k-1
                    flux_km = s.rho_w[k-1, i] * (-theta[k+1, i] + T(5.0) * theta[k, i] + T(2.0) * theta[k-1, i]) / T(6.0)
                else
                    flux_km = s.rho_w[k-1, i] * f2h(theta[k-1, i], theta[k, i], p.dz[k-1], p.dz[k])
                end
            end
            advection = -((flux_ip - flux_im) / dx + (flux_kp - flux_km) / p.dz[k]) # at (k, i)
            R_rho_theta[k, i] = advection + d_rho_theta_phys[k, i]
        end
    end

    # Difference from the previous state
    #rho_p = similar(s.rho); fill!(rho_p, T(0.0))
    #rho_u_p = similar(s.rho_u); fill!(rho_u_p, T(0.0))
    #rho_w_p = similar(s.rho_w); fill!(rho_w_p, T(0.0))
    #rho_theta_p = similar(s.rho_theta); fill!(rho_theta_p, T(0.0))
    rho_p = s0.rho .- s.rho
    rho_u_p = s0.rho_u .- s.rho_u
    rho_w_p = s0.rho_w .- s.rho_w
    rho_theta_p = s0.rho_theta .- s.rho_theta
    exchange_halo!(rho_theta_p, p)
    rho_theta_p_old = copy(rho_theta_p)

    # --- For save momentum flux for scalar advection ---
    mom_flux_x = copy(s.rho_u)
    mom_flux_z = copy(s.rho_w)

    # Short time steps
    dt_s = dt / ns
    for n in 1:ns
        #println("Step $n of $ns: dt_s = $dt_s")

        if n > 1
            exchange_halo!(rho_theta_p, p)
        end

        # --- Update rho_u at x-faces (k,i+1/2) ---
        Threads.@threads for i in p.is:p.ie
            #@turbo for k in p.ks:p.ke
            for k in p.ks:p.ke
                grad_x = (rho_theta_p[k, i+1] - rho_theta_p[k, i]) / dx # at (k,i+1/2)
                grad_x_old = (rho_theta_p_old[k, i+1] - rho_theta_p_old[k, i]) / dx
                grad_x = grad_x + gamma_d * (grad_x - grad_x_old) # Divergence damping
                rt2pres_h = (rt2pres[k, i] + rt2pres[k, i+1]) * T(0.5) # at (k,i+1/2)
                rho_u_p[k, i] += dt_s * (R_rho_u[k, i] - rt2pres_h * grad_x)
            end
        end
        #println("minmax: rho_u_p: ", minimum(rho_u_p[p.ks:p.ke, p.is:p.ie]), " ", maximum(rho_u_p[p.ks:p.ke, p.is:p.ie]))
        exchange_halo!(rho_u_p, p)


        rho_theta_p_old .= rho_theta_p # save value at the previous timestep for divergence damping
        fact_tp = T(0.5) * (1.0 + beta_offcenter)
        # --- Update rho_w at z-faces (k+1/2,i) (explicit part) ---
        Threads.@threads for i in p.is:p.ie
            # for off-centering
            rho_p_tp = similar(rho_p, p.ke)
            rho_theta_p_tp = similar(rho_theta_p, p.ke)
            @turbo for k in p.ks:p.ke
                #for k in p.ks:p.ke
                # rho
                advection = -(rho_u_p[k, i] - rho_u_p[k, i-1]) / dx # at (k,i)
                inc = dt_s * (R_rho[k, i] + advection)
                rho_p_tp[k] = rho_p[k, i] + fact_tp * inc
                rho_p[k, i] += inc # explicit

                # rho_theta
                flux_ip = (theta[k, i+1] + theta[k, i]) * T(0.5) * rho_u_p[k, i] # at (k,i+1/2)
                flux_im = (theta[k, i] + theta[k, i-1]) * T(0.5) * rho_u_p[k, i-1] # at (k,i-1/2)
                advection = -(flux_ip - flux_im) / dx # at (k,i)
                inc = dt_s * (R_rho_theta[k, i] + advection)
                rho_theta_p_tp[k] = rho_theta_p[k, i] + fact_tp * inc
                rho_theta_p[k, i] += inc # explicit
            end

            drho_w_p = similar(rho_w_p, p.ka)
            @turbo for k in p.ks:p.ke-1
                #for k in p.ks:p.ke-1
                buoyancy = (rho_theta_p_tp[k+1] - rho_theta_p_tp[k]) / (p.z[k+1] - p.z[k]) # at (k+1/2,i)
                grav = -GRAV * T(0.5) * (rho_p_tp[k+1] + rho_p_tp[k]) # at (k+1/2,i)
                rt2pres_h = (rt2pres[k, i] + rt2pres[k+1, i]) * T(0.5) # at (k+1/2,i)
                drho_w_p[k] = dt_s * (R_rho_w[k, i] - rt2pres_h * buoyancy + grav)
            end

            @views begin
                implicit_correction!(rho_w_p[:, i], drho_w_p, theta[:, i], rt2pres[:, i], dt_s, p) # Implicit correction step
            end

            # --- Update rho at centers (k,i) ---
            for k in p.ks:p.ke
                flux_kp = k == p.ke ? T(0.0) : rho_w_p[k, i]
                flux_km = k == p.ks ? T(0.0) : rho_w_p[k-1, i]
                advection = -(flux_kp - flux_km) / p.dz[k] # at (k,i)
                rho_p[k, i] += dt_s * advection
            end

            for k in p.ks:p.ke
                # Flux at (k+1/2,i)
                if k == p.ke
                    flux_kp = T(0.0)
                else
                    if rho_w_p[k, i] > EPS && k > p.ks # use k-1, k, k+1
                        flux_kp = rho_w_p[k, i] * (-theta[k-1, i] + T(5.0) * theta[k, i] + T(2.0) * theta[k+1, i]) / T(6.0)
                    elseif s.rho_w[k, i] < -EPS && k < p.ke - 1 # use k+2, k+1, k
                        flux_kp = rho_w_p[k, i] * (-theta[k+2, i] + T(5.0) * theta[k+1, i] + T(2.0) * theta[k, i]) / T(6.0)
                    else
                        flux_kp = rho_w_p[k, i] * f2h(theta[k, i], theta[k+1, i], p.dz[k], p.dz[k+1])
                    end
                end
                # Flux at (k-1/2,i)
                if k == p.ks
                    flux_km = T(0.0)
                else
                    if s.rho_w[k-1, i] > EPS && k > p.ks + 1 # use k-2, k-1, k
                        flux_km = rho_w_p[k-1, i] * (-theta[k-2, i] + T(5.0) * theta[k-1, i] + T(2.0) * theta[k, i]) / T(6.0)
                    elseif s.rho_w[k-1, i] < -EPS && k < p.ke # use k+1, k, k-1
                        flux_km = rho_w_p[k-1, i] * (-theta[k+1, i] + T(5.0) * theta[k, i] + T(2.0) * theta[k-1, i]) / T(6.0)
                    else
                        flux_km = rho_w_p[k-1, i] * f2h(theta[k-1, i], theta[k, i], p.dz[k-1], p.dz[k])
                    end
                end
                advection = -(flux_kp - flux_km) / p.dz[k] # at (k,i)
                rho_theta_p[k, i] += dt_s * advection
            end
        end
        #println("minmax: rho_theta_p: ", minimum(rho_theta_p[p.ks:p.ke, p.is:p.ie]), " ", maximum(rho_theta_p[p.ks:p.ke, p.is:p.ie]))


        # --- Momentum flux for scalar advection ---
        Threads.@threads for i in p.is:p.ie
            @turbo for k in p.ks:p.ke
                mom_flux_x[k, i] += rho_u_p[k, i] / ns
            end
            @turbo for k in p.ks:p.ke-1
                mom_flux_z[k, i] += rho_w_p[k, i] / ns
            end
        end

    end # short time steps

    exchange_halo!(mom_flux_x, p)
    exchange_halo!(s.rho_qv, p)
    exchange_halo!(s.rho_qc, p)
    exchange_halo!(s.rho_qr, p)

    # --- Scalar Transport Equations ---
    Threads.@threads for i in p.is:p.ie
        for k in p.ks:p.ke

            # --- Scalar Advection & Diffusion (rho * scalar) ---
            for (rho_scalar, rho_scalar0, d_phys) in [
                (s.rho_qv, s0.rho_qv, d_rho_qv_phys),
                (s.rho_qc, s0.rho_qc, d_rho_qc_phys),
                (s.rho_qr, s0.rho_qr, d_rho_qr_phys)]
                # Calculate fluxes at faces using helper functions
                # Face i+1/2
                flux_scalar_x_ip = compute_limited_flux_x(rho_scalar, s.rho, mom_flux_x, k, i, p)
                # Face i-1/2
                flux_scalar_x_im = compute_limited_flux_x(rho_scalar, s.rho, mom_flux_x, k, i - 1, p)
                adv_x = (flux_scalar_x_ip - flux_scalar_x_im) / dx

                # Face k+1/2
                flux_scalar_z_kp = k == p.ke ? FT(0.0) : compute_limited_flux_z(rho_scalar, s.rho, mom_flux_z, k, i, p)
                # Face k-1/2
                flux_scalar_z_km = k == p.ks ? FT(0.0) : compute_limited_flux_z(rho_scalar, s.rho, mom_flux_z, k - 1, i, p)
                adv_z = (flux_scalar_z_kp - flux_scalar_z_km) / p.dz[k]

                advection = -(adv_x + adv_z) # Negative divergence


                # Diffusion term: div(kappa * grad(scalar)) at cell center (k, i)
                # Using standard Finite Volume Method (Laplacian form)

                scalar_c = rho_scalar[k, i] / s.rho[k, i]
                scalar_ip = rho_scalar[k, i+1] / s.rho[k, i+1]
                scalar_im = rho_scalar[k, i-1] / s.rho[k, i-1]
                scalar_kp = k == p.ke ? T(0.0) : rho_scalar[k+1, i] / s.rho[k+1, i]
                scalar_km = k == p.ks ? T(0.0) : rho_scalar[k-1, i] / s.rho[k-1, i]

                # Flux difference in x
                dscalar_dx_ip = (s.rho[k, i+1] + s.rho[k, i]) * T(0.5) * (scalar_ip - scalar_c) / dx # rho * Gradient at (k,i+1/2)
                dscalar_dx_im = (s.rho[k, i] + s.rho[k, i-1]) * T(0.5) * (scalar_c - scalar_im) / dx # rho * Gradient at (k,i-1/2)
                diff_x = kappa * (dscalar_dx_ip - dscalar_dx_im) / dx

                # Flux difference in z
                dscalar_dz_kp = k == p.ke ? T(0.0) : f2h(s.rho[k+1, i], s.rho[k, i], p.dz[k+1], p.dz[k]) * (scalar_kp - scalar_c) / (p.z[k+1] - p.z[k]) # rho * Gradient at (k+1/2,i)
                dscalar_dz_km = k == p.ks ? T(0.0) : f2h(s.rho[k, i], s.rho[k-1, i], p.dz[k], p.dz[k-1]) * (scalar_c - scalar_km) / (p.z[k] - p.z[k-1]) # rho * Gradient at (k-1/2,i)
                diff_z = kappa * (dscalar_dz_kp - dscalar_dz_km) / p.dz[k]

                diffusion = diff_x + diff_z # Laplacian * kappa

                # Assign tendency
                rho_scalar[k, i] = rho_scalar0[k, i] + dt * (advection + diffusion + d_phys[k, i])
            end
        end
    end


    # --- Update rho, rho_u, rho_w, and rho_theta ---
    Threads.@threads for i in p.is:p.ie
        @turbo for k in p.ks:p.ke
            #s.rho[k, i] = s0.rho[k, i] + rho_p[k, i]
            #s.rho_u[k, i] = s0.rho_u[k, i] + rho_u_p[k, i]
            #s.rho_theta[k, i] = s0.rho_theta[k, i] + rho_theta_p[k, i]
            s.rho[k, i] = s.rho[k, i] + rho_p[k, i]
            s.rho_u[k, i] = s.rho_u[k, i] + rho_u_p[k, i]
            s.rho_theta[k, i] = s.rho_theta[k, i] + rho_theta_p[k, i]
            #s.rho[k, i] = s0.rho[k, i] + dt * R_rho[k, i]
            #s.rho_u[k, i] = s0.rho_u[k, i] + dt * R_rho_u[k, i]
            #s.rho_theta[k, i] = s0.rho_theta[k, i] + dt * R_rho_theta[k, i]
        end
        @turbo for k in p.ks:p.ke-1
            #s.rho_w[k, i] = s0.rho_w[k, i] + rho_w_p[k, i]
            s.rho_w[k, i] = s.rho_w[k, i] + rho_w_p[k, i]
            #s.rho_w[k, i] = s0.rho_w[k, i] + dt * R_rho_w[k, i]
        end
    end


    # check NaN values
    if any(isnan.(s.rho[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in rho")
    end
    if any(isnan.(s.rho_u[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in rho_u")
    end
    if any(isnan.(s.rho_w[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in rho_w")
    end
    if any(isnan.(s.rho_theta[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in rho_theta")
    end
    if any(isnan.(s.rho_qv[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in rho_qv")
    end
    if any(isnan.(s.rho_qc[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in rho_qc")
    end
    if any(isnan.(s.rho_qr[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in rho_qr")
    end
    if any(isnan.(d_rho_u_phys[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in d_rho_u_phys")
    end
    if any(isnan.(d_rho_theta_phys[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in d_rho_theta_phys")
    end
    if any(isnan.(d_rho_qv_phys[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in d_rho_qv_phys")
    end
    if any(isnan.(d_rho_qc_phys[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in d_rho_qc_phys")
    end
    if any(isnan.(d_rho_qr_phys[p.ks:p.ke, p.is:p.ie]))
        error("NaN detected in d_rho_qr_phys")
    end

end

# ============================================================
# 7. RK3 (Wicher and Skamarock 2002)
# Stage 1: v1 = v0 + f(v0) * dt / 3
# Stage 2: v2 = v0 + f(v1) * dt / 2
# Stage 3: v3 = v0 + f(v2) * dt
# Final output: v3
# ============================================================
function rk3_step!(s::State, p::Params{T}) where {T}
    dt = p.dt

    # --- Cloud Microphysics ---
    if p.do_mp
        cloud_microphysics!(s.d_rho_qv_mp, s.d_rho_qc_mp, s.d_rho_qr_mp, s.d_rho_theta_mp,
            s.rho, s.rho_theta, s.rho_qv, s.rho_qc, s.rho_qr, p)
    else
        fill!(s.d_rho_qv_mp, T(0.0))
        fill!(s.d_rho_qc_mp, T(0.0))
        fill!(s.d_rho_qr_mp, T(0.0))
        fill!(s.d_rho_theta_mp, T(0.0))
    end

    # --- Rain Sedimentation ---
    d_prec = device(similar(s.prec))
    if p.do_mp && p.do_sed
        rain_sedimentation!(s.d_rho_qr_sed, d_prec, s.rho_qr, s.rho, p)
    else
        fill!(s.d_rho_qr_sed, T(0.0))
        fill!(d_prec, T(0.0))
    end

    # --- Surface Fluxes ---
    if p.do_sf
        surface_fluxes!(s.d_rho_u_sf, s.d_rho_theta_sf, s.d_rho_qv_sf, s.mflux, s.shflux, s.lhflux, s.rho, s.rho_u, s.rho_theta, s.rho_qv, s.rho_qc, s.rho_qr, p)
    else
        fill!(s.d_rho_u_sf, T(0.0))
        fill!(s.d_rho_theta_sf, T(0.0))
        fill!(s.d_rho_qv_sf, T(0.0))
        fill!(s.mflux, T(0.0))
        fill!(s.shflux, T(0.0))
        fill!(s.lhflux, T(0.0))
    end

    # --- Radiative Cooling ---
    if p.do_rd
        #radiative_cooling!(d_rho_theta_rd, s.rho, s.rho_theta, s.rho_qv, s.rho_qc, s.rho_qr, p, rad_cooling)
        gray_radiation!(s.d_rho_theta_rd, s.rho, s.rho_theta, s.rho_qv, s.rho_qc, s.rho_qr, s.theta_ref, p)
    else
        fill!(s.d_rho_theta_rd, T(0.0))
    end

    # --- Physical Source/Sink Terms ---
    d_rho_phys = copy(s.d_rho_qr_sed)
    d_rho_phys[p.ks, :] .+= s.d_rho_qv_sf
    d_rho_u_phys = device(similar(s.rho_u))
    fill!(d_rho_u_phys, T(0.0))
    d_rho_u_phys[p.ks, :] .+= s.d_rho_u_sf
    d_rho_theta_phys = @. s.d_rho_theta_mp + s.d_rho_theta_rd + d_rho_phys * s.rho_theta / s.rho
    d_rho_theta_phys[p.ks, :] .+= s.d_rho_theta_sf
    d_rho_qv_phys = copy(s.d_rho_qv_mp)
    d_rho_qv_phys[p.ks, :] .+= s.d_rho_qv_sf
    d_rho_qc_phys = s.d_rho_qc_mp
    d_rho_qr_phys = @. s.d_rho_qr_mp + s.d_rho_qr_sed
    phys_args = (d_rho_phys, d_rho_u_phys, d_rho_theta_phys, d_rho_qv_phys, d_rho_qc_phys, d_rho_qr_phys)

    # --- RK3 Step ---
    s0 = deepcopy(s) # Save state for RK3
    # Stage 1
    ns = max(1, ceil(Int, p.ns / 3))
    compute_step!(phys_args..., s, s0, p, dt / 3, ns)

    # Stage 2
    ns = max(1, ceil(Int, p.ns / 2))
    compute_step!(phys_args..., s, s0, p, dt / 2, ns)

    # Stage 3
    compute_step!(phys_args..., s, s0, p, dt, p.ns)

    #=
    cloud_microphysics!(d_rho_qv_mp, d_rho_qc_mp, d_rho_qr_mp, d_rho_theta_mp,
        s.rho, s.rho_theta, s.rho_qv, s.rho_qc, s.rho_qr, p)
    rain_sedimentation!(d_rho_qr_sed, d_prec, s.rho_qr, s.rho, p)

        Threads.@threads for i in p.is:p.ie
        @turbo for k in p.ks:p.ke
            s.rho[k, i] += dt * d_rho_qr_sed[k, i]
            s.rho_theta[k, i] += dt * (d_rho_theta_mp[k, i] + d_rho_theta_rd[k, i] + d_rho_qr_sed[k, i] * s0.rho_theta[k, i] / s0.rho[k, i])
            s.rho_qv[k, i] += dt * d_rho_qv_mp[k, i]
            s.rho_qc[k, i] += dt * d_rho_qc_mp[k, i]
            s.rho_qr[k, i] += dt * (d_rho_qr_mp[k, i] + d_rho_qr_sed[k, i])
        end
        s.rho[p.ks, i] += dt * d_rho_qv_sf[i]
        s.rho_u[p.ks, i] += dt * d_rho_u_sf[i]
        s.rho_theta[p.ks, i] += dt * d_rho_theta_sf[i]
        s.rho_qv[p.ks, i] += dt * d_rho_qv_sf[i]
    end
    =#

    # Precipitation
    Threads.@threads for i in p.is:p.ie
        s.prec[i] += d_prec[i] * dt
    end
end

# NetCDF variables for history
@with_kw mutable struct History
    var_time
    var_rho
    var_rhou
    var_rhow
    var_u
    var_w
    var_theta
    var_qv
    var_qc
    var_qr
    var_RH
    var_temp
    var_pres
    var_thetae
    var_prec
    var_mflux
    var_shflux
    var_lhflux
    var_d_rho_theta_mp
    var_d_rho_qv_mp
    var_d_rho_qc_mp
    var_d_rho_qr_mp
    var_d_rho_qr_sed
    var_d_rho_u_sf
    var_d_rho_theta_sf
    var_d_rho_qv_sf
    var_d_rho_theta_rd
end

function output_history(hist::History, nc::NCDataset, t::Float64, n_out::Int, s::State, p::Params{T}, Etot0::Float64) where {T}

    # Calculate primitive variables for output
    exchange_halo!(s.rho_u, p)
    rho_int = s.rho[p.ks:p.ke, p.is:p.ie]
    rho_inv_int = @. T(1.0) / rho_int
    theta_int = s.rho_theta[p.ks:p.ke, p.is:p.ie] .* rho_inv_int
    qv_int = s.rho_qv[p.ks:p.ke, p.is:p.ie] .* rho_inv_int
    qc_int = s.rho_qc[p.ks:p.ke, p.is:p.ie] .* rho_inv_int
    qr_int = s.rho_qr[p.ks:p.ke, p.is:p.ie] .* rho_inv_int

    u_int = similar(rho_int)
    @views u_int .= @. (s.rho_u[p.ks:p.ke, p.is:p.ie] + s.rho_u[p.ks:p.ke, p.is-1:p.ie-1]) * T(0.5) * rho_inv_int
    w_int = similar(rho_int)
    @views w_int[2:p.Nz, :] .= @. (s.rho_w[p.ks+1:p.ke, p.is:p.ie] + s.rho_w[p.ks:p.ke-1, p.is:p.ie]) * T(0.5) * rho_inv_int[2:p.Nz, :]
    @views w_int[1, :] .= @. s.rho_w[p.ks, p.is:p.ie] * T(0.5) * rho_inv_int[1, :] # rho_w[ks-1, :] is zero

    pres_int = similar(rho_int)
    temp_int = similar(rho_int)
    thetae_int = similar(rho_int)
    RH_int = similar(rho_int)
    eng_k = Array{Float64}(undef, p.Nz, p.Nx) # Kinetic energy
    eng_p = similar(eng_k) # Potential energy
    eng_i = similar(eng_k) # Internal energy (moist)
    for i in 1:p.Nx
        for k in 1:p.Nz
            qt = qv_int[k, i] + qc_int[k, i] + qr_int[k, i]
            qd = T(1.0) - qt
            r = Rd * qd + Rv * qv_int[k, i]
            cp = Cpd * qd + Cpv * qv_int[k, i] + Cpl * (qc_int[k, i] + qr_int[k, i])
            cv = cp - r
            pi_ki = (r * s.rho_theta[p.ks+k-1, p.is+i-1] / P0)^(r / cv)
            pres_int[k, i] = P0 * pi_ki^(cp / r)
            temp_int[k, i] = theta_int[k, i] * pi_ki
            es_ki, _ = saturation_vapor_pressure(temp_int[k, i])
            qsat_ki, _ = saturation_specific_humidity(pres_int[k, i], es_ki)
            RH_int[k, i] = qsat_ki > FT(1e-10) ? qv_int[k, i] / qsat_ki * T(100.0) : T(0.0)

            # theta_e
            presv = rho_int[k, i] * qv_int[k, i] * Rv * temp_int[k, i] # partial pressure of vapor
            presd = pres_int[k, i] - presv # partial pressure of dry air
            cp = Cpd * qd + Cpl * qt
            thetae_int[k, i] = temp_int[k, i] * (presd / P0)^(-Rd / cp) * exp(Lv * qv_int[k, i] / (cp * temp_int[k, i]))

            # Energy
            eng_k[k, i] = T(0.5) * rho_int[k, i] * (u_int[k, i]^2 + w_int[k, i]^2)
            eng_p[k, i] = rho_int[k, i] * GRAV * p.z[k]
            eng_i[k, i] = rho_int[k, i] * (cv * temp_int[k, i] + qv_int[k, i] * Lv)
        end
    end

    wmax = maximum(w_int)
    wmin = minimum(w_int)
    qcmax = maximum(qc_int) * 1000 # g/kg
    qrmax = maximum(qr_int) * 1000 # g/kg
    rhmin = minimum(RH_int)
    rhmax = maximum(RH_int)
    eng_t = sum(eng_k) + sum(eng_p) + sum(eng_i)

    if Etot0 < 0.0
        Etot0 = eng_t
    end
    eng_t -= Etot0
    @info @sprintf("t=%.2f s | w(min,max)=(%.2f, %.2f) m/s | qcmax=%.2f g/kg | qrmax=%.2f g/kg | RH(min,max)=(%.2f, %.2f) %% | Etot=%.2e", t, wmin, wmax, qcmax, qrmax, rhmin, rhmax, eng_t)

    # Write to NetCDF
    hist.var_time[n_out] = t
    hist.var_rho[:, :, n_out] = rho_int'
    hist.var_rhou[:, :, n_out] = s.rho_u[p.ks:p.ke, p.is:p.ie]'
    hist.var_rhow[:, :, n_out] = s.rho_w[p.ks:p.ke, p.is:p.ie]'
    hist.var_u[:, :, n_out] = u_int'
    hist.var_w[:, :, n_out] = w_int'
    hist.var_theta[:, :, n_out] = theta_int'
    hist.var_qv[:, :, n_out] = qv_int'
    hist.var_qc[:, :, n_out] = qc_int'
    hist.var_qr[:, :, n_out] = qr_int'
    hist.var_RH[:, :, n_out] = RH_int'
    hist.var_temp[:, :, n_out] = temp_int'
    hist.var_pres[:, :, n_out] = pres_int' ./ T(100.0) # Pa -> hPa
    hist.var_thetae[:, :, n_out] = thetae_int'
    hist.var_prec[:, n_out] = s.prec[p.is:p.ie]
    hist.var_mflux[:, n_out] = s.mflux[p.is:p.ie]
    hist.var_shflux[:, n_out] = s.shflux[p.is:p.ie]
    hist.var_lhflux[:, n_out] = s.lhflux[p.is:p.ie]
    hist.var_d_rho_theta_mp[:, :, n_out] = s.d_rho_theta_mp[p.ks:p.ke, p.is:p.ie]'
    hist.var_d_rho_qv_mp[:, :, n_out] = s.d_rho_qv_mp[p.ks:p.ke, p.is:p.ie]'
    hist.var_d_rho_qc_mp[:, :, n_out] = s.d_rho_qc_mp[p.ks:p.ke, p.is:p.ie]'
    hist.var_d_rho_qr_mp[:, :, n_out] = s.d_rho_qr_mp[p.ks:p.ke, p.is:p.ie]'
    hist.var_d_rho_qr_sed[:, :, n_out] = s.d_rho_qr_sed[p.ks:p.ke, p.is:p.ie]
    hist.var_d_rho_u_sf[:, n_out] = s.d_rho_u_sf[p.is:p.ie]'
    hist.var_d_rho_theta_sf[:, n_out] = s.d_rho_theta_sf[p.is:p.ie]'
    hist.var_d_rho_qv_sf[:, n_out] = s.d_rho_qv_sf[p.is:p.ie]'
    hist.var_d_rho_theta_rd[:, :, n_out] = s.d_rho_theta_rd[p.ks:p.ke, p.is:p.ie]'
    sync(nc)

    return Etot0
end


# ============================================================
# 8. Main driver
# ============================================================
function run_sim(; p=Params()) # Use default constructor

    @info "Configuration: $(p.Nx) x $(p.Nz) | dt=$(p.dt) | dx=$(p.dx) "

    num_threads = Threads.nthreads()
    rngs = [Random.Xoshiro(i) for i in 1:num_threads]
    state = init_state!(allocate_state(p), p, rngs)
    #state = init_state_BF2002!(allocate_state(p), p)


    fill!(state.prec, FT(0.0)) # Initialize precipitation to zero


    for A in (state.rho, state.rho_u, state.rho_w, state.rho_theta, state.rho_qv, state.rho_qc, state.rho_qr)
        exchange_halo!(A, p)
    end


    # Prepare NetCDF file output
    nc_filename = "history.nc"
    isfile(nc_filename) && rm(nc_filename) # Remove existing file
    nc = NCDataset(nc_filename, "c")
    defDim(nc, "x", p.Nx)
    defDim(nc, "z", p.Nz)
    defDim(nc, "time", Inf)
    var_x = defVar(nc, "x", FT, ("x",))
    var_z = defVar(nc, "z", FT, ("z",))
    var_zf = defVar(nc, "zf", FT, ("z",))
    hist = History(
        var_time = defVar(nc, "time", FT, ("time",)),
        var_rho = defVar(nc, "rho", FT, ("x", "z", "time")),
        var_rhou = defVar(nc, "rhou", FT, ("x", "z", "time")),
        var_rhow = defVar(nc, "rhow", FT, ("x", "z", "time")),
        var_u = defVar(nc, "u", FT, ("x", "z", "time")), # velocity u
        var_w = defVar(nc, "w", FT, ("x", "z", "time")), # velocity w
        var_theta = defVar(nc, "theta", FT, ("x", "z", "time")), # theta
        var_qv = defVar(nc, "qv", FT, ("x", "z", "time")), # qv
        var_qc = defVar(nc, "qc", FT, ("x", "z", "time")), # qc
        var_qr = defVar(nc, "qr", FT, ("x", "z", "time")), # qr
        var_RH = defVar(nc, "RH", FT, ("x", "z", "time")), # Relative Humidity
        var_temp = defVar(nc, "temp", FT, ("x", "z", "time")), # temperature
        var_pres = defVar(nc, "pres", FT, ("x", "z", "time")), # pressure
        var_thetae = defVar(nc, "thetae", FT, ("x", "z", "time")), # equivalent potential temperature
        var_prec = defVar(nc, "prec", FT, ("x", "time")), # precipitation
        var_mflux = defVar(nc, "mflux", FT, ("x", "time")), # mass flux
        var_shflux = defVar(nc, "shflux", FT, ("x", "time")), # sensible heat flux
        var_lhflux = defVar(nc, "lhflux", FT, ("x", "time")), # latent heat flux
        var_d_rho_theta_mp = defVar(nc, "d_rho_theta_mp", FT, ("x", "z", "time")), # d_rho_theta_mp
        var_d_rho_qv_mp = defVar(nc, "d_rho_qv_mp", FT, ("x", "z", "time")), # d_rho_qv_mp
        var_d_rho_qc_mp = defVar(nc, "d_rho_qc_mp", FT, ("x", "z", "time")), # d_rho_qc_mp
        var_d_rho_qr_mp = defVar(nc, "d_rho_qr_mp", FT, ("x", "z", "time")), # d_rho_qr_mp
        var_d_rho_qr_sed = defVar(nc, "d_rho_qr_sed", FT, ("x", "z", "time")), # d_rho_qr_sed
        var_d_rho_u_sf = defVar(nc, "d_rho_u_sf", FT, ("x", "time")), # d_rho_u_sf
        var_d_rho_theta_sf = defVar(nc, "d_rho_theta_sf", FT, ("x", "time")), # d_rho_theta_sf
        var_d_rho_qv_sf = defVar(nc, "d_rho_qv_sf", FT, ("x", "time")), # d_rho_qv_sf
        var_d_rho_theta_rd = defVar(nc, "d_rho_theta_rd", FT, ("x", "z", "time")) # d_rho_theta_rd
    )

    # Add attributes (units)
    var_x.attrib["units"] = "m"
    var_z.attrib["units"] = "m"
    var_zf.attrib["units"] = "m"
    hist.var_time.attrib["units"] = "s"
    hist.var_rho.attrib["units"] = "kg/m^3"
    hist.var_rhou.attrib["units"] = "kg/m^2/s"
    hist.var_rhow.attrib["units"] = "kg/m^2/s"
    hist.var_u.attrib["units"] = "m/s"
    hist.var_w.attrib["units"] = "m/s"
    hist.var_theta.attrib["units"] = "K"
    hist.var_qv.attrib["units"] = "kg/kg"
    hist.var_qc.attrib["units"] = "kg/kg"
    hist.var_qr.attrib["units"] = "kg/kg"
    hist.var_RH.attrib["units"] = "%"
    hist.var_temp.attrib["units"] = "K"
    hist.var_pres.attrib["units"] = "hPa"
    hist.var_thetae.attrib["units"] = "K"
    hist.var_prec.attrib["units"] = "kg/m^2"
    hist.var_mflux.attrib["units"] = "kg/m/s^2"
    hist.var_shflux.attrib["units"] = "W/m^2"
    hist.var_lhflux.attrib["units"] = "W/m^2"
    hist.var_d_rho_theta_mp.attrib["units"] = "kg K/m^3/s"
    hist.var_d_rho_qv_mp.attrib["units"] = "kg/m^3/s"
    hist.var_d_rho_qc_mp.attrib["units"] = "kg/m^3/s"
    hist.var_d_rho_qr_mp.attrib["units"] = "kg/m^3/s"
    hist.var_d_rho_qr_sed.attrib["units"] = "kg/m^3/s"
    hist.var_d_rho_u_sf.attrib["units"] = "kg/m^2/s^2"
    hist.var_d_rho_theta_sf.attrib["units"] = "kg K/m^3/s"
    hist.var_d_rho_qv_sf.attrib["units"] = "kg/kg/m^3/s"
    hist.var_d_rho_theta_rd.attrib["units"] = "kg K/m^3/s"
 
    var_x[:] = p.x[p.is:p.ie]
    var_z[:] = p.z[p.ks:p.ke]
    var_zf[:] = p.zf[p.ks:p.ke]

    t = Float64(0.0)
    Etot0::Float64 = -1.0
    n_out = 1
    Etot0 = output_history(hist, nc, t, n_out, state, p, Etot0)

    @info "Running for $(p.N_steps) steps"
    for it in 1:p.N_steps
        rk3_step!(state, p)

        if mod(it, p.Nint_steps) == 0
            t = Float64(it * p.dt)

            # --- Check for NaN ---
            if any(isnan, state.rho) || any(isnan, state.rho_theta) || any(isnan, state.rho_qv)
                @error "NaN detected at step $it, time $t. Aborting."
                close(nc)
                return state # Return current state for inspection
            end

            n_out += 1
            output_history(hist, nc, t, n_out, state, p, Etot0)

        end
    end

    close(nc)
    @info "Simulation completed. Output saved to $nc_filename"
    return state
end

# script entry
if abspath(PROGRAM_FILE) == @__FILE__
    # Make sure to start Julia with threads enabled, e.g., julia -t auto your_script.jl
    if Threads.nthreads() == 1
        @warn "Julia started with 1 thread. Run with 'julia -t auto your_script.jl' for parallel execution."
    end
    run_sim()
end
