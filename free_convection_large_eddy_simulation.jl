using Oceananigans, Oceananigans.Units, GLMakie

# Specify domain and mesh / grid
Nx = Ny = Nz = 128
Lx = Ly = 128
Lz = 64

grid = RectilinearGrid(GPU(), size = (Nx, Ny, Nz), halo = (3, 3, 3),
                       x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

# Specify coriolis, viscosity / turbulence closure, and boundary condition
coriolis = FPlane(f₀=1e-4)
closure = AnisotropicMinimumDissipation()
boundary_conditions = (; b = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-8)))

model = NonhydrostaticModel(; grid, coriolis, closure, boundary_conditions,
                              advection = WENO5(),
                              tracers = :b,
                              buoyancy = BuoyancyTracer())
                          
# Time-stepping
simulation = Simulation(model, Δt=10minutes, stop_time=4hours)

progress(sim) = @info "Iter: $(iteration(sim)), time: $(prettytime(sim))"
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

wizard = TimeStepWizard(cfl=0.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

run!(simulation)

# Visualize the x-velocity field
fig, ax, pl = heatmap(interior(model.velocities.w)[:, 1, :])
display(fig)
