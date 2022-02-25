using Oceananigans, Oceananigans.Units, GLMakie

grid = RectilinearGrid(CPU(), size = (64, 64, 64), halo = (3, 3, 3),
                       x = (0, 128), y = (0, 128), z = (-64, 0),
                       topology = (Periodic, Periodic, Bounded))

# Specify coriolis, viscosity / turbulence closure, and boundary condition
coriolis = FPlane(f₀=1e-4)
closure = AnisotropicMinimumDissipation()
boundary_conditions = (; b = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-8)))

#=
parameters = (μ₀ = 1/day,  # surface growth rate
              λ = 5,       # sunlight attenuation length scale (m)
              m = 0.1/day) # mortality rate due to virus and zooplankton grazing
@inline growing_and_grazing(x, y, z, t, P, params) = (params.μ₀ * exp(z / params.λ) - params.m) * P
plankton_dynamics = Forcing(growing_and_grazing; field_dependencies = :P, parameters)
tracers = (:b, :P)
forcing = (; P = plankton_dynamics)
=#
tracers = :b

model = NonhydrostaticModel(; grid, coriolis, closure, boundary_conditions, tracers,
                              timestepper = :RungeKutta3, advection = WENO5(),
                              buoyancy = BuoyancyTracer())
                          
bᵢ(x, y, z) = 1e-6 * z
ϵ(x, y, z) = 1e-6 * randn() # noise
set!(model, b=bᵢ, u=ϵ, v=ϵ, w=ϵ)

# Time-stepping
simulation = Simulation(model, Δt=10minutes, stop_time=12hours)

progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|w|: %s\n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.w))

wizard = TimeStepWizard(cfl=0.8, max_change=1.1, max_Δt=2minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Turbulent kinetic energy
u, v, w = model.velocities
e = Field((u^2 + v^2 + w^2) / 2)
ξ = Field(∂x(w) + ∂z(u))

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; e, w, ξ, b = model.tracers.b),
                                                      schedule = TimeInterval(1hour),
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
