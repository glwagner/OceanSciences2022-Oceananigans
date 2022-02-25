using Statistics
using JLD2
using Printf
# using GLMakie
using Oceananigans
using Oceananigans.Units

using Oceananigans.Architectures: arch_array

using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization

#####
##### Grid
#####

latitude = (-84.375, 84.375)
Δφ = latitude[2] - latitude[1]

# 2.8125 degree resolution
Nx = 128
Ny = 60
Nz = 8

output_prefix = "annual_cycle_global_lat_lon_$(Nx)_$(Ny)_$(Nz)_temp"

arch = CPU()
reference_density = 1035

#####
##### Load forcing files 
#####

using DataDeps

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/lat_lon_bathymetry_and_fluxes/"

dh = DataDep("near_global_lat_lon",
    "Forcing data for global latitude longitude simulation",
    [path * "bathymetry_lat_lon_128x60_FP32.bin",
     path * "sea_surface_temperature_25_128x60x12.jld2",
     path * "tau_x_128x60x12.jld2",
     path * "tau_y_128x60x12.jld2"]
)

DataDeps.register(dh)
datadep"near_global_lat_lon"

filename = [:sea_surface_temperature_25_128x60x12, :tau_x_128x60x12, :tau_y_128x60x12]

for name in filename
    datadep_path = @datadep_str "near_global_lat_lon/" * string(name) * ".jld2"
    file = Symbol(:file_, name)
    @eval $file = jldopen($datadep_path)
end

bathymetry = Array{Float32}(undef, Nx*Ny)
bathymetry_path = @datadep_str "near_global_lat_lon/bathymetry_lat_lon_128x60_FP32.bin"
read!(bathymetry_path, bathymetry)

bathymetry = bswap.(bathymetry) |> Array{Float64}
bathymetry = arch_array(arch, reshape(bathymetry, Nx, Ny))

τˣ = zeros(Nx, Ny)
τʸ = zeros(Nx, Ny)
T★ = zeros(Nx, Ny)

τˣ = arch_array(arch, file_tau_x_128x60x12["tau_x/1"] ./ reference_density)
τʸ = arch_array(arch, file_tau_y_128x60x12["tau_y/1"] ./ reference_density)
T★ = arch_array(arch, file_sea_surface_temperature_25_128x60x12["sst25/1"])

H = 3600.0

# A spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch,
                                              size = (Nx, Ny, Nz),
                                              longitude = (-180, 180),
                                              latitude = latitude,
                                              halo = (3, 3, 3),
                                              z = (-H, 0),
                                              precompute_metrics = true)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

#####
##### Physics and model setup
#####

νh₄ = 1e+13
νz₂ = 1e+1
κh₂ = 1e+3
κz₂ = 1e-4

vertical_closure       = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = νz₂, κ = κz₂)
horizontal_viscosity   = HorizontalScalarBiharmonicDiffusivity(ν = νh₄)
horizontal_diffusivity = HorizontalScalarDiffusivity(κ = κh₂)

background_diffusivity = (horizontal_viscosity, horizontal_diffusivity, vertical_closure)
convective_adjustment  = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

#####
##### Boundary conditions / constant-in-time surface forcing
#####

# Surface temperature relaxation parameter [ms⁻¹]
const λ = grid.Δzᵃᵃᶠ / 3days  

@inline function surface_temperature_relaxation(i, j, grid, clock, fields, p)

    T★        = p.T★[i, j]
    T_surface = fields.T[i, j, grid.Nz]             
    
    return p.λ * (T_surface - T★)
end

T_surface_relaxation_bc = FluxBoundaryCondition(surface_temperature_relaxation,
                                                discrete_form = true,
                                                parameters = (λ = λ, T★ = T★))

@inline wind_stress(i, j, grid, clock, fields, τ) = τ[i, j]

u_wind_stress_bc = FluxBoundaryCondition(wind_stress, discrete_form = true, parameters = - τˣ)
v_wind_stress_bc = FluxBoundaryCondition(wind_stress, discrete_form = true, parameters = - τʸ)

u_bcs = FieldBoundaryConditions(top = u_wind_stress_bc)
v_bcs = FieldBoundaryConditions(top = v_wind_stress_bc)
T_bcs = FieldBoundaryConditions(top = T_surface_relaxation_bc)

free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver)

model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                    momentum_advection = VectorInvariant(),
                                    tracer_advection = WENO5(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs),
                                    buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=true),
                                    tracers = (:T, ),
                                    closure = (background_diffusivity..., convective_adjustment)) 

#####
##### Initial condition:
#####

model.tracers.T .= -1

#####
##### Simulation setup
#####

# Realistic time step
Δt = 20minutes

simulation = Simulation(model, Δt = Δt, stop_time = 5years)

progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# Let's goo!
run!(simulation)

####
#### Visualize solution
####

# using GLMakie

# fig = Figure(resolution = (1200, 900))

# ax = Axis(fig[1, 1], title="Free surface displacement (m)")
# hm = GLMakie.heatmap!(ax, η[:, :, 1], colorrange=(min_η, max_η), colormap=:balance)
# cb = Colorbar(fig[1, 2], hm)

# ax = Axis(fig[2, 1], title="Sea surface temperature (ᵒC)")
# hm = GLMakie.heatmap!(ax, T[:, :, Nx], colorrange=(min_T, max_T), colormap=:thermal)
# cb = Colorbar(fig[2, 2], hm)

# ax = Axis(fig[1, 3], title="East-west surface velocity (m s⁻¹)")
# hm = GLMakie.heatmap!(ax, u[:, :, Nx], colorrange=(min_u, max_u), colormap=:balance)
# cb = Colorbar(fig[1, 4], hm)

# ax = Axis(fig[2, 3], title="North-south surface velocity (m s⁻¹)")
# hm = GLMakie.heatmap!(ax, v[:, :, Nx], colorrange=(min_u, max_u), colormap=:balance)
# cb = Colorbar(fig[2, 4], hm)

# display(fig)

