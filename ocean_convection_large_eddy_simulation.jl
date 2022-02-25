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
                          
bᵢ(x, y, z) = 1e-5 * z
ϵ(x, y, z) = 1e-6 * randn() # noise
set!(model, b=bᵢ, u=ϵ, v=ϵ, w=ϵ)

# Time-stepping
simulation = Simulation(model, Δt=10minutes, stop_time=24hours)

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

