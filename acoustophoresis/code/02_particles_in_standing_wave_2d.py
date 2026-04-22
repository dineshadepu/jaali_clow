import numpy as np
import matplotlib.pyplot as plt

def write_structured_vtk(filename, X, Y, U, Fx, Fy):
    nx, ny = X.shape

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Acoustic field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write(f"POINTS {nx*ny} float\n")

        for j in range(ny):
            for i in range(nx):
                f.write(f"{X[i,j]} {Y[i,j]} 0.0\n")

        f.write(f"\nPOINT_DATA {nx*ny}\n")

        # scalar field
        f.write("SCALARS U float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{U[i,j]}\n")

        # vector field
        f.write("\nVECTORS F float\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{Fx[i,j]} {Fy[i,j]} 0.0\n")


def write_particles_vtk(filename, xp, yp):
    n = len(xp)

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Particles\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")

        for i in range(n):
            f.write(f"{xp[i]} {yp[i]} 0.0\n")


# -----------------------------
# Physical parameters (REALISTIC)
# -----------------------------
rho0 = 1000.0        # density (kg/m^3)
c0   = 1500.0        # speed of sound (m/s)
k    = 8378.0        # wave number (1/m)

E0   = 10.0          # energy density (J/m^3)
p0   = np.sqrt(4 * rho0 * c0**2 * E0)  # pressure amplitude

# contrast factors (choose material)
# polystyrene:
f0 = 0.46
f1 = 0.038

# silicone oil (alternative):
# f0 = -0.08
# f1 = 0.07

# coefficients (IMPORTANT: includes k)
A = f0 / (3 * rho0**2 * (c0 * k)**2)
B = f1 / 2.0

# -----------------------------
# Fluid + particle properties
# -----------------------------
mu = 1e-3        # viscosity (Pa.s)
a  = 12e-6       # particle radius (m)

gamma = 6 * np.pi * mu * a

# -----------------------------
# Domain
# -----------------------------
nx, ny = 150, 150
Lx, Ly = 1e-3, 1e-3   # 1 mm domain

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# -----------------------------
# Standing wave fields
# -----------------------------
p2 = 0.5 * p0**2 * (np.cos(k*X)**2) * (np.cos(k*Y)**2)

v2 = 0.5 * (p0/(rho0*c0))**2 * (
      (np.sin(k*X)**2)*(np.cos(k*Y)**2)
    + (np.cos(k*X)**2)*(np.sin(k*Y)**2)
)

# -----------------------------
# Radiation potential
# -----------------------------
U = A*p2 - B*v2

# -----------------------------
# Force = -grad(U)
# -----------------------------
dx = x[1] - x[0]
dy = y[1] - y[0]

dU_dx, dU_dy = np.gradient(U, dx, dy)

Fx = -dU_dx
Fy = -dU_dy


# -----------------------------
# Data export to vtk
# -----------------------------
write_structured_vtk("acoustic_field.vtk", X, Y, U, Fx, Fy)

# import pyvista as pv

# # Create grid
# grid = pv.StructuredGrid(X, Y, np.zeros_like(X))

# # Add scalar field
# grid["U"] = U.flatten(order="F")

# # Add vector field
# vectors = np.stack([Fx, Fy, np.zeros_like(Fx)], axis=-1)
# grid["F"] = vectors.reshape(-1, 3, order="F")

# grid.save("acoustic_field.vtk")

# -----------------------------
# Particle initialization
# -----------------------------
n_particles = 25
xp = np.random.rand(n_particles) * Lx
yp = np.random.rand(n_particles) * Ly

# -----------------------------
# Interpolation (nearest grid)
# -----------------------------
def get_force(xp, yp):
    ix = np.clip(((xp - x[0]) / (x[-1] - x[0]) * (nx-1)).astype(int), 0, nx-1)
    iy = np.clip(((yp - y[0]) / (y[-1] - y[0]) * (ny-1)).astype(int), 0, ny-1)
    return Fx[iy, ix], Fy[iy, ix]

# -----------------------------
# Time stepping
# -----------------------------
dt = 1e-4
nsteps = 500

traj_x = [xp.copy()]
traj_y = [yp.copy()]

for t in range(nsteps):

    Fx_p, Fy_p = get_force(xp, yp)

    vx = Fx_p / gamma
    vy = Fy_p / gamma

    xp += vx * dt
    yp += vy * dt

    xp = np.clip(xp, 0, Lx)
    yp = np.clip(yp, 0, Ly)

    traj_x.append(xp.copy())
    traj_y.append(yp.copy())

    # >>> ADD EXPORT HERE <<<
    if t % 10 == 0:
        write_particles_vtk(f"particles_{t:04d}.vtk", xp, yp)

# # -----------------------------
# # Plot
# # -----------------------------
# plt.figure(figsize=(8,6))

# # potential field
# plt.contourf(X, Y, U, levels=50)
# plt.colorbar(label="Radiation potential U")

# # force field
# skip = 6
# plt.quiver(X[::skip,::skip], Y[::skip,::skip],
#            Fx[::skip,::skip], Fy[::skip,::skip],
#            color='white', scale=5e-11)

# # trajectories
# for i in range(n_particles):
#     plt.plot([traj_x[t][i] for t in range(len(traj_x))],
#              [traj_y[t][i] for t in range(len(traj_y))],
#              'w-', linewidth=1)

# # final positions
# plt.scatter(xp, yp, color='cyan', s=25, label="Particles")

# plt.title("Acoustic Particle Trapping (Standing Wave)")
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.legend()
# plt.tight_layout()
# plt.show()
