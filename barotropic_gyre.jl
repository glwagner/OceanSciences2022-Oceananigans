using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: Horizontal
using GLMakie

grid = RectilinearGrid(size = (60, 60, 1),
                       x = (0, 1200kilometers),
                       y = (0, 1200kilometers),
                       z = (-4kilometers, 0),
                       halo = (3, 3, 3),
                       topology = (Bounded, Bounded, Bounded))

# Specify coriolis, viscosity / turbulence closure, and boundary condition
coriolis = BetaPlane(f₀=1e-4, β=1e-11)
closure = ScalarDiffusivity(ν=100, isotropy=Horizontal())

no_slip = ValueBoundaryCondition(0)
wind_stress(x, y, t) = - 1e-4 * cos(2π * x / 1200kilometers)
boundary_conditions = (;
    u = FieldBoundaryConditions(top=FluxBoundaryCondition(wind_stress), south=no_slip, north=no_slip),
    v = FieldBoundaryConditions(east=no_slip, west=no_slip)
)

model = HydrostaticFreeSurfaceModel(; grid, coriolis, closure, boundary_conditions,
                                      momentum_advection = UpwindBiasedFifthOrder(),
                                      tracers = (), buoyancy = nothing)
                          
# Time-stepping
simulation = Simulation(model, Δt=20seconds, stop_time=5days)

progress(sim) = @info "Iter: $(iteration(sim)), time: $(prettytime(sim))"
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

run!(simulation)

# Visualize the x-velocity field
fig, ax, pl = heatmap(interior(model.velocities.u)[:, :, 1])
display(fig)
