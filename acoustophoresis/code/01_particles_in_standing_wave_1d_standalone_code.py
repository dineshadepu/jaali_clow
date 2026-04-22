import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Physical parameters
# -----------------------------
p0   = 1.0        # pressure amplitude
rho0 = 1.0        # density
c0   = 1.0        # sound speed
k    = 2*np.pi    # wave number

# acoustic contrast factors (play with these!)
f0 = 1.0
f1 = 1.0

# coefficients
A = f0 / (3 * rho0**2 * c0**2)
B = f1 / 2.0

# -----------------------------
# Spatial domain
# -----------------------------
x = np.linspace(0, 2, 500)

# -----------------------------
# Time-averaged fields
# -----------------------------
p2 = 0.5 * p0**2 * np.cos(k*x)**2
v2 = 0.5 * (p0/(rho0*c0))**2 * np.sin(k*x)**2

# -----------------------------
# Radiation potential
# -----------------------------
U = A * p2 - B * v2

# -----------------------------
# Force = -grad(U)
# -----------------------------
dx = x[1] - x[0]
F = -np.gradient(U, dx)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(x, p2, label=r"$\langle p^2 \rangle$")
plt.plot(x, v2, label=r"$\langle v^2 \rangle$")
plt.plot(x, U,  label=r"$U(x)$")
plt.plot(x, F,  label=r"$F(x)$")

plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.title("Standing wave: nodes, potential, and radiation force")
plt.xlabel("x")
plt.grid(True)

plt.show()
