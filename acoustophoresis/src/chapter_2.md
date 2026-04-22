# Validation Setup: From Energy Density to Physical Forces

## 1. What We Are Computing

Because we **prescribe a standing wave**, the first-order acoustic fields are already known:

$$
p_1(x,t), \quad \mathbf{v}_1(x,t)
$$

From these we compute **time-averaged quadratic quantities**:

$$
\langle p_1^2 \rangle, \quad \langle v_1^2 \rangle
$$

Then build the **radiation potential**:

$$
U = A \langle p_1^2 \rangle - B \langle v_1^2 \rangle
$$

and finally the radiation force:

$$
\mathbf{F}^{\mathrm{rad}}_{\mathrm{ext}} = -\nabla U
$$

The workflow is therefore:

> Given an acoustic field → compute energy landscape → particles move along its gradient

---

## 2. Primary vs Secondary Radiation Force

We compute only the **primary (external) radiation force**:

- Arises from interaction with the imposed standing wave
- Responsible for trapping at nodes or antinodes

We do **not** include the **secondary radiation force**:

- Arises from particle–particle interactions via scattered waves
- Responsible for clustering and pattern formation between particles

For single-particle validation, only the primary force is needed.

---

## 3. Why Trapping Appears

For a 1D standing wave:

$$
p \sim \cos(kx), \quad v \sim \sin(kx)
$$

The time-averaged quadratic quantities become:

$$
\langle p^2 \rangle \sim \cos^2(kx), \quad \langle v^2 \rangle \sim \sin^2(kx)
$$

This creates a spatially periodic energy landscape with minima at either nodes or antinodes depending on the contrast factors $f_0$ and $f_1$. Since:

$$
\mathbf{F} = -\nabla U
$$

particles slide toward the minima of $U$. This is the origin of acoustic trapping.

---

## 4. Parameters from the Paper

| Symbol | Value | Units |
| ------ | ----- | ----- |
| $\rho_0$ | $1000$ | kg/m³ |
| $c_0$ | $1500$ | m/s |
| $f$ | $2$ | MHz |
| $k$ | $8378$ | m⁻¹ |
| $E_0$ | $10$ | J/m³ |
| $a$ | $12$ | µm |

### Contrast factors

| Material | $f_0$ | $f_1$ | Trapping site |
| -------- | ----- | ----- | ------------- |
| Silicone oil | $-0.08$ | $0.07$ | Pressure antinode |
| Polystyrene | $0.46$ | $0.038$ | Pressure node |

---

## 5. Getting Pressure Amplitude from Energy Density

The paper specifies the acoustic energy density $E_0$, but our force formula requires the pressure amplitude $p_0$. This is the key link.

Acoustic energy density is:

$$
E_0 = \frac{\langle p^2 \rangle}{2 \rho_0 c_0^2}
$$

For a harmonic wave, $\langle p^2 \rangle = p_0^2 / 2$, so:

$$
E_0 = \frac{p_0^2}{4 \rho_0 c_0^2}
$$

Solving for $p_0$:

$$
\boxed{p_0 = \sqrt{4 \rho_0 c_0^2 E_0}}
$$

Plugging in the paper's values:

$$
p_0 = \sqrt{4 \cdot 1000 \cdot (1500)^2 \cdot 10} \approx 3 \times 10^5 \, \text{Pa}
$$

This is approximately **300 kPa**, which is realistic for acoustofluidics.

---

## 6. Full Simulation Setup

```python
rho0 = 1000.0          # kg/m^3
c0   = 1500.0          # m/s
k    = 8378.0          # m^-1  (NOT 2*pi — see note below)

E0   = 10.0            # J/m^3
p0   = (4*rho0*c0**2*E0)**0.5   # ~3e5 Pa

a    = 12e-6           # m
mu   = 1e-3            # Pa·s  (water viscosity)
```

> **Important:** Use `k = 8378.0`, not `k = 2*pi`. The toy value `k = 2*pi` breaks physical scaling and will give forces off by many orders of magnitude.

---

## 7. Expected Validation Result

From the paper, the radiation force magnitude for a single particle is approximately **1 pN**:

$$
F_{\mathrm{rad}} \sim 10^{-12} \, \text{N}
$$

### Quick order-of-magnitude check

$$
a^3 \sim 10^{-15}, \quad E_0 \sim 10, \quad k \sim 10^4
$$

$$
F \sim a^3 k E_0 \sim 10^{-15} \cdot 10^4 \cdot 10 = 10^{-10}
$$

After scaling by contrast factors $f_0, f_1 \ll 1$, this falls into the **pN range** — consistent with the paper.

---

## 8. What to Observe in Simulation

### Polystyrene ($f_0 = 0.46 > 0$)

Particles migrate toward **pressure nodes** where $\cos(kx) = 0$.

### Silicone oil ($f_0 = -0.08 < 0$)

Particles migrate toward **pressure antinodes** where $\cos(kx) = \pm 1$.

This sign flip of $f_0$ is the physical mechanism behind **selective trapping** of different particle types in the same acoustic field.

---

## 9. Validation Checklist

Before comparing with the paper:

- [ ] Use the physical wave number $k = 8378 \, \text{m}^{-1}$, not $2\pi$
- [ ] Derive pressure amplitude from energy density: $p_0 = \sqrt{4 \rho_0 c_0^2 E_0}$
- [ ] Verify that the coefficient $A$ scales with $(\rho_0 \omega)^{-2}$, which carries the $k^2$ dependence
- [ ] Check that force magnitude falls in the **pN range** ($\sim 10^{-12}$ N)
- [ ] Confirm polystyrene particles collect at pressure **nodes**
- [ ] Confirm oil droplets collect at pressure **antinodes**

If the force is off by orders of magnitude, the cause is almost always one of:

1. Wrong $p_0$ (forgot to convert from $E_0$)
2. Missing $k^2$ in the coefficient $A$

---

## 10. Summary

| Step | Formula |
| ---- | ------- |
| Prescribe wave | $p(x,t) = p_0 \cos(kx)\cos(\omega t)$ |
| Time-average | $\langle p^2 \rangle = \frac{p_0^2}{2}\cos^2(kx)$ |
| Build potential | $U = A\langle p^2 \rangle - B\langle v^2 \rangle$ |
| Compute force | $F = -\nabla U$ |
| Move particles | $v_p = F / (6\pi\mu a)$ |
| Convert $E_0 \to p_0$ | $p_0 = \sqrt{4\rho_0 c_0^2 E_0}$ |

$$
\boxed{
\text{Use real wave fields, time-averaged quantities, and physical } k \text{ to reach pN-scale forces}
}
$$
