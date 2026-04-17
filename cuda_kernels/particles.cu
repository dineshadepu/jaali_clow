#define M_PI 3.14159265358979323846

// ---------------------------------------------------------------------------
// Field gather
// ---------------------------------------------------------------------------

// For each particle, copy the cell-centred field values into per-particle
// arrays.  Particles outside the mesh (tid < 0) get zeros.
extern "C" __global__
void gather_cell_data(
    const int*    triangle_ids,
    const double* cell_vel_x,
    const double* cell_vel_y,
    const double* cell_pressure,
    double*       p_vel_x,
    double*       p_vel_y,
    double*       p_pressure,
    int           n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    int tid = triangle_ids[i];
    if (tid < 0) {
        p_vel_x[i]    = 0.0;
        p_vel_y[i]    = 0.0;
        p_pressure[i] = 0.0;
        return;
    }
    p_vel_x[i]    = cell_vel_x[tid];
    p_vel_y[i]    = cell_vel_y[tid];
    p_pressure[i] = cell_pressure[tid];
}

// ---------------------------------------------------------------------------
// Tracer advection  (particle velocity == fluid velocity, no inertia)
// ---------------------------------------------------------------------------

// pos += dt * fluid_vel
extern "C" __global__
void advect_tracer_2d(
    double*       px,
    double*       py,
    const double* fluid_vel_x,
    const double* fluid_vel_y,
    double        dt,
    int           n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    px[i] += dt * fluid_vel_x[i];
    py[i] += dt * fluid_vel_y[i];
}

// ---------------------------------------------------------------------------
// Periodicity
// ---------------------------------------------------------------------------

// Wrap positions into [x_min, x_max) x [y_min, y_max).
extern "C" __global__
void wrap_periodic_2d(
    double* px,
    double* py,
    double  x_min, double x_max,
    double  y_min, double y_max,
    int     n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    double lx = x_max - x_min;
    double ly = y_max - y_min;

    px[i] = x_min + fmod(fmod(px[i] - x_min, lx) + lx, lx);
    py[i] = y_min + fmod(fmod(py[i] - y_min, ly) + ly, ly);
}

// ---------------------------------------------------------------------------
// Force integrator  (compute forces → update velocity → update position)
// ---------------------------------------------------------------------------

// Stokes drag: F = -6*pi*mu*radius * (v_particle - v_fluid)
extern "C" __global__
void stokes_drag_2d(
    const double* fluid_vel_x,
    const double* fluid_vel_y,
    const double* vel_x_p,
    const double* vel_y_p,
    double*       force_x_p,
    double*       force_y_p,
    double        mu,
    double        radius,
    int           n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    double gamma = 6.0 * M_PI * mu * radius;
    force_x_p[i] = -gamma * (vel_x_p[i] - fluid_vel_x[i]);
    force_y_p[i] = -gamma * (vel_y_p[i] - fluid_vel_y[i]);
}

// vel_p += dt * force_p / mass_p
extern "C" __global__
void update_velocity_2d(
    double*       vel_x_p,
    double*       vel_y_p,
    const double* force_x_p,
    const double* force_y_p,
    const double* mass_p,
    double        dt,
    int           n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    double inv_m = 1.0 / mass_p[i];
    vel_x_p[i] += dt * force_x_p[i] * inv_m;
    vel_y_p[i] += dt * force_y_p[i] * inv_m;
}

// pos += dt * vel_p
extern "C" __global__
void update_position_2d(
    double*       px,
    double*       py,
    const double* vel_x_p,
    const double* vel_y_p,
    double        dt,
    int           n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    px[i] += dt * vel_x_p[i];
    py[i] += dt * vel_y_p[i];
}
