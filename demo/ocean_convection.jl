using Oceananigans

# Idealized convection in a rectilinear in x-z
grid = RectilinearGrid(size=(64, 64), x=(0, 128), z=(-64, 0), halo=(3, 3),
                       topology=(Periodic, Flat, Bounded))

# Build a model
top_buoyancy_bc = FluxBoundaryCondition(1e-8)
buoyancy_bcs = FieldBoundaryConditions(top=top_buoyancy_bc)

closure = ScalarDiffusivity(ν=1e-4, κ=1e-4)

model = NonhydrostaticModel(grid = grid,
                            advection = WENO5(),
                            tracers = :b,
                            closure = closure,
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (; b=buoyancy_bcs))

N² = 1e-6
bᵢ(x, y, z) = N² * z
ϵ(x, y, z) = 1e-6 * randn()
set!(model, b=bᵢ, u=ϵ, w=ϵ)

# Set up simulation
simulation = Simulation(model, Δt=2minutes, stop_time=2hours)

progress(sim) = @printf("Iter: %d, time: %s \n", iteration(sim), prettytime(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

run!(simulation)


