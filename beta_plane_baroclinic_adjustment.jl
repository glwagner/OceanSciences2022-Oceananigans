using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Printf
using GLMakie
using JLD2

grid = RectilinearGrid(CPU(),
                       topology = (Periodic, Bounded, Bounded), 
                       size = (128, 128, 16),
                       x = (-500kilometers, 500kilometers),
                       y = (-500kilometers, 500kilometers),
                       z = (-4kilometers, 0),
                       halo = (3, 3, 3))

const Lz = grid.Lz

# Uncomment to put a bump in the grid:
# This will slow down the simulation because we will use a
# matrix-based Poisson solver instead of the FFT-based solver.

const width = 50kilometers
bump(x, y) = - Lz * (1 - 0.5 * exp(-(x^2 + y^2) / 2width^2))
# grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bump))

# Physics

Δx = grid.Lx / grid.Nx
κ₄h = Δx^4 / 1day
κz = 1e-2

diffusive_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=κz, κ=κz)
horizontal_closure = HorizontalScalarBiharmonicDiffusivity(ν=κ₄h, κ=κ₄h)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = BetaPlane(latitude = -45),
                                    buoyancy = BuoyancyTracer(),
                                    closure = (diffusive_closure, horizontal_closure),
                                    tracers = (:b, :c),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface())

# Initial condition: a baroclinically unstable situation!
ramp(y, δy) = min(max(0, y/δy + 1/2), 1)

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

δy = 50kilometers
δz = 400
Lz = grid.Lz

δc = 2δy
δb = δy * M²
ϵb = 1e-2 * δb # noise amplitude

bᵢ(x, y, z) = N² * z + δb * ramp(y, δy) + ϵb * randn()
cᵢ(x, y, z) = exp(-y^2 / 2δc^2) * exp(-(z + Lz/4)^2 / 2δz^2)

set!(model, b=bᵢ, c=cᵢ)

simulation = Simulation(model, Δt=10minutes, stop_time=60days)

wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=simulation.Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

print_progress(sim) =
    @printf("Iter: %d, time: %s, wall time: %s, max|u|: %6.3e, m s⁻¹, next Δt: %s\n",
            iteration(sim), prettytime(sim), prettytime(sim.run_wall_time),
            maximum(abs, sim.model.velocities.u), prettytime(sim.Δt))

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))

b, c = model.tracers
u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))

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

maxζ = maximum(maximum(abs, ζi) for ζi in ζ)
maxb = maximum(maximum(bi) for bi in b)
minb = minimum(minimum(bi) for bi in b)

Nt = length(t)

fig = Figure(resolution=(1800, 600))

axζ = Axis(fig[1, 1])
axb = Axis(fig[1, 2])
axc = Axis(fig[1, 3])

slider = Slider(fig[3, :], range=1:Nt, startvalue=Nt)
n = slider.value

xζ, yζ, zζ = 1e3 .* nodes((Face, Face, Center), grid)
xc, yc, zc = 1e3 .* nodes((Center, Center, Center), grid)

ζⁿ = @lift ζ[$n]
bⁿ = @lift b[$n]
cⁿ = @lift c[$n]

hmζ = heatmap!(axζ, xζ, yζ, ζⁿ, colormap=:redblue, colorrange=(-maxζ, maxζ))
hmb = heatmap!(axb, xc, yc, bⁿ, colormap=:thermal, colorrange=(minb, maxb))
hmc = heatmap!(axc, xc, yc, cⁿ, colormap=:deep, colorrange=(0, 0.3))

Colorbar(fig[2, 1], hmζ, vertical=false, flipaxis=true, label="Vertical vorticity (s⁻¹)")
Colorbar(fig[2, 2], hmb, vertical=false, flipaxis=true, label="Buoyancy (m s⁻²)")
Colorbar(fig[2, 3], hmc, vertical=false, flipaxis=true, label="Tracer concentration")

title = @lift "Baroclinic adjustment at t = " * prettytime(t[$n])
Label(fig[0, :], title)

display(fig)

# record(fig, "beta_plane_baroclinic_adjustment.mp4", 1:Nt, framerate=12) do nn
#     @info "Rendering frame $nn of $Nt..."
#     n[] = nn
# end
