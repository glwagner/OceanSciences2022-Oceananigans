using Oceananigans
using Oceananigans.Units
using GLMakie
using Printf

grid = RectilinearGrid(size=(64, 64), x=(0, 128), z=(-64, 0), halo=(3, 3),
                       topology=(Periodic, Flat, Bounded))

boundary_conditions = (; b = FieldBoundaryConditions(top = FluxBoundaryCondition(1e-8)))
closure = ScalarDiffusivity(ν=1e-4, κ=1e-4)
coriolis = FPlane(f=1e-4)

model = NonhydrostaticModel(; grid, closure, coriolis, boundary_conditions,
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer())

bᵢ(x, y, z) = 1e-6 * z
ϵ(x, y, z) = 1e-6 * randn() # noise
set!(model, b=bᵢ, u=ϵ, v=ϵ, w=ϵ)

simulation = Simulation(model, Δt=2minutes, stop_time=12hours)

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
b = FieldTimeSeries(data_path, "b")
x, y, z = nodes(e)
Nt = length(e.times)

fig = Figure(resolution=(1600, 800))
axe = Axis(fig[1, 1])
axξ = Axis(fig[1, 2])
slider = Slider(fig[3, :], range=1:Nt, startvalue=1)
n = slider.value

eⁿ = @lift dropdims(interior(e[$n]), dims=2)
bⁿ = @lift dropdims(interior(b[$n]), dims=2)
hm = heatmap!(axe, x, z, eⁿ; colormap=:solar, colorrange=(0, maximum(e) / 4))
Colorbar(fig[2, 1], hm, vertical=false, flipaxis=false, label="Turbulent kinetic energy (m² s⁻²)")

hm = heatmap!(axξ, x, z, bⁿ; colormap=:deep, colorrange=(minimum(b), maximum(b)))
Colorbar(fig[2, 2], hm, vertical=false, flipaxis=false, label="Buoyancy (m s⁻²)")

title = @lift "Idealized ocean convection at t = " * prettytime(e.times[$n])
Label(fig[0, :], title)

display(fig)

#record(fig, "idealized_ocean_convection.mp4", 1:Nt, framerate=12) do nn
#    @info "Rendering frame $nn of $Nt...
#    n[] = nn
#end
