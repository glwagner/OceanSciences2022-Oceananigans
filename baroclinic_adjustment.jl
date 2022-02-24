using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.TurbulenceClosures: VerticallyImplicit, Vertical, Horizontal
using Printf
using GLMakie
using JLD2

grid = RectilinearGrid(topology = (Periodic, Bounded, Bounded), 
                       size = (128, 128, 16),
                       x = (-500kilometers, 500kilometers),
                       y = (-500kilometers, 500kilometers),
                       z = (-1kilometers, 0),
                       halo = (3, 3, 3))

const Lz = grid.Lz

# Uncomment to put a bump in the grid:
#=
const width = 50kilometers
bump(x, y) = - Lz * (1 - 0.5 * exp(-x^2 / 2width^2))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bump))
=#

Δx = grid.Lx / grid.Nx
κh = Δx^4 / 12hours

diffusive_closure = ScalarDiffusivity(ν = 1e-2, κ = 1e-2,
                                      isotropy = Vertical(),
                                      time_discretization = VerticallyImplicit())

horizontal_closure = ScalarBiharmonicDiffusivity(ν = κh, κ = κh, isotropy = Horizontal())

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
δz = 100
Lz = grid.Lz

δc = 2δy
δb = δy * M²
ϵb = 1e-2 * δb # noise amplitude

bᵢ(x, y, z) = N² * z + δb * ramp(y, δy) + ϵb * randn()
cᵢ(x, y, z) = exp(-y^2 / 2δc^2) * exp(-(z + Lz/4)^2 / 2δz^2)

set!(model, b=bᵢ, c=cᵢ)

simulation = Simulation(model, Δt=20minutes, stop_time=30days)

wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=simulation.Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(sim.run_wall_time),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    return nothing
end

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

Nt = length(t)

fig = Figure()

axζ = Axis(fig[1, 1])
axb = Axis(fig[1, 2])
axc = Axis(fig[1, 3])

slider = Slider(fig[2, :], range=1:Nt, startvalue=1)
n = slider.value

xζ, yζ, zζ = 1e3 .* nodes((Face, Face, Center), grid)
xc, yc, zc = 1e3 .* nodes((Center, Center, Center), grid)

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
