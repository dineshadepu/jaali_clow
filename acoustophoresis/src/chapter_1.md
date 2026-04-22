# Acoustic Particle Forces: From Theory to Implementation

## 1. Governing Acoustic Field (Paper Formulation)

The paper starts from a **harmonic acoustic field** using the velocity potential:

$$
\Phi(\mathbf{r}, t) = \phi(\mathbf{r}) e^{-i \omega t}
$$

which satisfies the Helmholtz equation:

$$
\nabla^2 \phi(\mathbf{r}) = -k^2 \phi(\mathbf{r}), \quad k = \frac{\omega}{c_0}
$$

From this potential, the first-order acoustic fields are:

$$
p_1(\mathbf{r}) = i \omega \rho_0 , \phi(\mathbf{r})
$$

$$
\rho_1(\mathbf{r}) = i \frac{\omega \rho_0}{c_0^2} , \phi(\mathbf{r})
$$

$$
\mathbf{v}_1(\mathbf{r}) = \nabla \phi(\mathbf{r})
$$

---

## 2. Radiation Force (Single Particle)

The radiation force is derived from a potential:

$$
\mathbf{F}^{rad}_{ext} = -\nabla U(\mathbf{r})
$$

where

$$ U(\mathbf{r}) = \frac{\epsilon_p^3 \pi \rho_0}{k} \left[\frac{f_0}{3} |\phi(\mathbf{r})|^2 - \frac{f_1}{2} |\nabla \phi(\mathbf{r})|^2 \right] $$

---

# 3. What This Means Physically

* **Primary radiation force** → interaction with external wave
* **Secondary radiation force** → interaction via scattered waves

---

# 4. What We Changed (Key Implementation Shift)

The paper works in **complex harmonic form** using:

$$
\phi(\mathbf{r}) \in \mathbb{C}
$$

However, in our implementation:

---

## ❗ We do NOT solve for $\phi$ explicitly

Instead, we:

1. **Prescribe pressure field analytically**
2. Work directly with **real-valued fields**

---
# 4.1 Why We Can Avoid $\phi$

Even though the paper is formulated in terms of the velocity potential $\phi$,
the radiation force depends only on **quadratic, time-averaged quantities**:

$$
|\phi|^2 \quad \text{and} \quad |\nabla \phi|^2
$$

Using the relations:

$$
p_1 = i \omega \rho_0 \phi, \quad
\mathbf{v}_1 = \nabla \phi
$$

we can rewrite everything in terms of **measurable real fields**:

$$
|\phi|^2 \propto \langle p_1^2 \rangle, \quad
|\nabla \phi|^2 \propto \langle v_1^2 \rangle
$$

Thus, we completely eliminate $\phi$ and work directly with:

$$
p, \quad \mathbf{v}
$$

# 5. Our Assumed Acoustic Field

We assume a **standing wave**:

$$
p(x,t) = p_0 \cos(kx)\cos(\omega t)
$$

From this:

$$
v(x,t) = \frac{p_0}{\rho_0 c_0} \sin(kx)\sin(\omega t)
$$

---

# 6. Time Averaging (Key Step)

The radiation force depends on **time-averaged quadratic terms**:

$$
\langle p_1^2 \rangle = \frac{p_0^2}{2} \cos^2(kx)
$$

$$
\langle v_1^2 \rangle = \frac{p_0^2}{2 \rho_0^2 c_0^2} \sin^2(kx)
$$

---

# 7. Equivalent Radiation Potential (Our Form)

Instead of using $\phi$, we rewrite:

$$
U(x) = A \langle p_1^2 \rangle - B \langle v_1^2 \rangle
$$

where:

$$
A = \frac{f_0}{3 \omega^2 \rho_0^2}, \quad
B = \frac{f_1}{2}
$$

---

# 8. Radiation Force (Final Computable Form)

$$
\mathbf{F}^{rad} = -\nabla U
$$

---

## 💻 Discrete form

```text
p2 = average(p*p)
v2 = average(u*u + v*v)

U = A*p2 - B*v2

Fx = -dU/dx
Fy = -dU/dy
```

---

# 9. Particle Dynamics (Overdamped Regime)

At microscale:

$$
6\pi \mu a , \mathbf{v}_p = \mathbf{F}^{rad}
$$

So:

$$
\mathbf{v}_p = \frac{\mathbf{F}^{rad}}{6\pi \mu a}
$$

---

## 💻 Update rule

```text
vx = Fx / (6*pi*mu*a)
vy = Fy / (6*pi*mu*a)

x += vx * dt
y += vy * dt
```

---

# 10. Physical Interpretation of the Force

For a standing wave:

$$
U(x) \propto \cos^2(kx) \quad \text{and} \quad \sin^2(kx)
$$

Thus:

- Pressure nodes: $\cos(kx) = 0$
- Velocity nodes: $\sin(kx) = 0$

Particles move toward:

- **Pressure nodes** if $f_0 > 0$
- **Pressure antinodes** if $f_0 < 0$

This is the origin of acoustic trapping.

# 11. Summary of Differences

| Paper                        | Implementation                |
| ---------------------------- | ----------------------------- |
| Uses complex $\phi$          | Uses real $p, v$              |
| Works in frequency domain    | Works in time-averaged domain |
| Exact scattering formulation | Approximate pairwise force    |
| Analytical gradients         | Numerical gradients           |

---

# 13. Final Pipeline

1. Define standing wave:
   $$
   p(x,t), v(x,t)
   $$

2. Compute averages:
   $$
   \langle p^2 \rangle, \langle v^2 \rangle
   $$

3. Compute potential:
   $$
   U = A p^2 - B v^2
   $$

4. Compute force:
   $$
   F = -\nabla U
   $$

5. Move particles:
   $$
   v_p = \frac{F}{6\pi\mu a}
   $$

6. (Optional) add interactions:
   $$
   F_{int}
   $$

---

# 🔥 Final Insight

$$
\boxed{
\text{We replaced complex harmonic acoustics with real, time-averaged fields}
}
$$

$$
\boxed{
\text{This makes the model computationally tractable while preserving physics}
}
$$

---
