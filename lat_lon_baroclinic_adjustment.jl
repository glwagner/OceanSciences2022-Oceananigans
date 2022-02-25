using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.TurbulenceClosures: VerticallyImplicit, Vertical, Horizontal
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField
using Printf
using GLMakie
using JLD2

grid = LatitudeLongitudeGrid(size = (64, 1024, 4),
                             longitude = (-180, 180),
                             latitude = (30, 50),
                             z = (-1kilometers, 0),
                             halo = (3, 3, 3))

const Lz = grid.Lz

# Uncomment to put a bump in the grid:
#=
const width = 10 # degrees
bump(λ, φ) = - Lz * (1 - 0.5 * exp(-λ^2 / 2width^2))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bump))
=#

diffusive_closure = ScalarDiffusivity(ν = 1e-2, κ = 1e-2,
                                      isotropy = Vertical(),
                                      time_discretization = VerticallyImplicit())

horizontal_closure = ScalarBiharmonicDiffusivity(ν=1e6, κ=1e6, isotropy = Horizontal())

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    buoyancy = BuoyancyTracer(),
                                    closure = (diffusive_closure, horizontal_closure),
                                    tracers = (:b, :c),
                                    momentum_advection = VectorInvariant(),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface())

# Initial condition: a baroclinically unstable situation!
ramp(φ, δφ, φ₀) = min(max(0, (φ - φ₀)/δφ + 1/2), 1)

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

φ₀ = 40 # degrees
δφ = 1 # degrees
δz = 400 # m
δc = 2δφ
δb = δφ * M²
ϵb = 1e-2 * δb # noise amplitude

bᵢ(λ, φ, z) = N² * z + δb * ramp(φ, δφ, φ₀) + ϵb * randn()
cᵢ(λ, φ, z) = exp(-(φ - φ₀)^2 / 2δc^2) * exp(-(z + Lz/4)^2 / 2δz^2)

set!(model, b=bᵢ, c=cᵢ)

simulation = Simulation(model, Δt=20minutes, stop_time=30days)

print_progress(sim) = @printf("i: %d, t: %s, wall time: %s, max(u): %6.3e m s⁻¹ \n",
                              iteration(sim), prettytime(sim), prettytime(sim.run_wall_time),
                              maximum(abs, sim.model.velocities.u))

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))

b, c = model.tracers
u, v, w = model.velocities
ζ = VerticalVorticityField(model)

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ζ, b, c),
                                                      schedule = TimeInterval(1hour),
                                                      field_slicer = FieldSlicer(k=grid.Nz),
                                                      prefix = "baroclinic_adjustment",
                                                      force = true)

run!(simulation)

file = jldopen("baroclinic_adjustment.jld2")
iterations = parse.(Int, keys(file["timeseries/t"]))
ζ = [file["timeseries/ζ/$i"][:, :, 1] for i in iterations]
b = [file["timeseries/b/$i"][:, :, 1] for i in iterations]
c = [file["timeseries/c/$i"][:, :, 1] for i in iterations]
t = [file["timeseries/t/$i"] for i in iterations]
close(file)

Nt = length(t)

fig = Figure()

axζ = Axis(fig[1, 1], aspect=:data)
axb = Axis(fig[2, 1], aspect=:data)
axc = Axis(fig[3, 1], aspect=:data)

slider = Slider(fig[2, :], range=1:Nt, startvalue=1)
n = slider.value

xζ, yζ, zζ = nodes((Face, Face, Center), grid)
xc, yc, zc = nodes((Center, Center, Center), grid)

ζⁿ = @lift ζ[$n]
bⁿ = @lift b[$n]
cⁿ = @lift c[$n]

hmζ = heatmap!(axζ, xζ, yζ, ζⁿ)
hmb = heatmap!(axb, xc, yc, bⁿ)
hmc = heatmap!(axc, xc, yc, cⁿ)

# Colorbar(fig[2, 1], xζ, yζ, hmζ, horizontal=true)
# Colorbar(fig[2, 2], xc, yc, hmb, horizontal=true)
# Colorbar(fig[2, 3], xc, yc, hmc, horizontal=true)

display(fig)

