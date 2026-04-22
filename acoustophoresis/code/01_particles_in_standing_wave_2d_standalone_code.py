import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
p0   = 1.0
rho0 = 1.0
c0   = 1.0
k    = 2*np.pi

# contrast factors (change these!)
f0 = 1.0
f1 = 1.0

A = f0 / (3 * rho0**2 * c0**2)
B = f1 / 2.0

# -----------------------------
# Grid
# -----------------------------
nx, ny = 150, 150
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# -----------------------------
# Time-averaged fields
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
# Plot
# -----------------------------
plt.figure(figsize=(8,6))

# background = potential
plt.contourf(X, Y, U, levels=50)
plt.colorbar(label="Radiation potential U")

# quiver (force field)
skip = 6
plt.quiver(X[::skip,::skip], Y[::skip,::skip],
           Fx[::skip,::skip], Fy[::skip,::skip],
           color='white', scale=50)

plt.title("2D Acoustic Radiation Force Field")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()

plt.show()
