using Oceananigans
using Oceananigans.Units
using GLMakie
using Printf

grid = RectilinearGrid(size=(64, 64), extent=(64, 64), halo=(3, 3), topology=(Periodic, Flat, Bounded))
buoyancy_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(1e-8))

model = NonhydrostaticModel(grid = grid, 
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                            coriolis = FPlane(f=1e-4),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (; b=buoyancy_bcs))

bᵢ(x, y, z) = 1e-5 * z
ϵ(x, y, z) = 1e-6 * randn() # noise
set!(model, b=bᵢ, u=ϵ, v=ϵ, w=ϵ)

simulation = Simulation(model, Δt=2minutes, stop_time=24hours)

wizard = TimeStepWizard(cfl=0.8, max_change=1.1, max_Δt=2minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|w|: %s\n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.w))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# Turbulent kinetic energy
u, v, w = model.velocities
e = Field((u^2 + v^2 + w^2) / 2)
ξ = Field(∂x(w) + ∂z(u))

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; e, w, ξ, b = model.tracers.b),
                                                      schedule = TimeInterval(1minute),
                                                      prefix = "ocean_convection",
                                                      field_slicer = nothing,
                                                      force = true)

run!(simulation)

# Visualization

data_path = simulation.output_writers[:fields].filepath
e = FieldTimeSeries(data_path, "e")
x, y, z = nodes(e)
Nt = length(e.times)

fig = Figure()
ax = Axis(fig[1, 1])
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value

eⁿ = @lift dropdims(interior(e[$n]), dims=2)
hm = heatmap!(ax, x, z, eⁿ; colormap=:solar, colorrange=(0, maximum(e) / 2))
Colorbar(fig[1, 2], hm)

title = @lift "Turbulent kinetic energy at t = " * prettytime(e.times[$n])
Label(fig[0, :], title)

display(fig)

