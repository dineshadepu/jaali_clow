# jaali_clow

GPU-accelerated 2-D particle simulation library for acoustic and Stokes flow problems.
Built on CUDA via [cudarc](https://github.com/coreylowman/cudarc).

---

## Requirements

| Tool | Version |
|------|---------|
| Rust | stable (edition 2024) |
| CUDA toolkit | 13.0.1 |
| `nvcc` | matching your CUDA installation |
| Python | 3.8+ (tools only) |

---

## Building

```bash
# Compile CUDA kernels to PTX first (required before `cargo build`)
nvcc -ptx -O2 cuda_kernels/locate_triangles.cu -o cuda_kernels/locate_triangles.ptx
nvcc -ptx -O2 cuda_kernels/particles.cu        -o cuda_kernels/particles.ptx

# Build the library
cargo build --release
```

> **Architecture flag**: If `nvcc` defaults to an arch that doesn't match your GPU, add
> `-arch=sm_XX` (e.g. `-arch=sm_86` for Ampere). Check the existing PTX header for the
> arch the kernels were last compiled for:
> ```bash
> head -2 cuda_kernels/locate_triangles.ptx
> ```

---

## Repository layout

```
jaali_clow/
├── src/
│   ├── lib.rs             — crate root; re-exports locator and particles
│   ├── locator.rs         — Locator2D: BVH-based triangle locator
│   ├── particles.rs       — FluidField2D and ParticleAdvector2D
│   ├── filtered_mesh.rs   — StokesField, FilteredMesh, filter_mesh, find_body_bounds
│   ├── io.rs              — VTK readers/writers and ParaView helper scripts
│   ├── bvh.rs             — CPU BVH builder, GPU upload
│   ├── mesh.rs            — TriMesh, TriMeshGPU
│   └── gpu/               — CudaManager, CudaStream, CudaSlice, CudaFunction wrappers
│
├── cuda_kernels/
│   ├── locate_triangles.cu/.ptx  — BVH traversal kernels
│   └── particles.cu/.ptx         — particle physics kernels
│
├── examples/
│   ├── particles_in_stokes_2d.rs                       — tracer advection in Stokes flow
│   ├── acoustic_particle_trapping_2d.rs                — tracer advection in acoustic field
│   └── acoustic_particle_trapping_2d_01_only_stokes_drag.rs  — Stokes-drag particle integrator
│
└── tools/
    └── find_period.py     — detect acoustic period from VTK time-series
```

---

## Library modules

### `locator` — `Locator2D`

Locates each query point inside a triangle mesh using a GPU BVH.

```rust
use jaali_clow::locator::Locator2D;
use jaali_clow::mesh::TriMesh;

let mesh = TriMesh { vx: &vx, vy: &vy, t0: &t0, t1: &t1, t2: &t2 };
let mut locator = Locator2D::new(&mesh)?;

// ids_d[i] = triangle index containing particle i, or -1 if outside mesh
locator.locate(&px_d, &py_d, &mut ids_d)?;
```

Reads kernels from `cuda_kernels/locate_triangles.ptx`.

---

### `particles` — `FluidField2D` and `ParticleAdvector2D`

**`FluidField2D`** holds cell-centred fluid data (velocity, pressure) and gathers
interpolated values onto per-particle arrays after each locator step.

```rust
use jaali_clow::particles::FluidField2D;

let mut fluid = FluidField2D::new(stream.clone(), &vel_x, &vel_y, &pressure)?;

// After locator.locate(...):
fluid.gather(&ids_d)?;                       // triangle IDs → per-particle fields

// Tracer advection (massless particles that just follow the flow):
fluid.advect_tracer(&mut px_d, &mut py_d, dt)?;
fluid.apply_periodicity(&mut px_d, &mut py_d, x_lo, x_hi, y_lo, y_hi)?;
```

**`ParticleAdvector2D`** integrates inertial particles using a force → velocity →
position scheme.  Construct it once, then call in order each time step:

```rust
use jaali_clow::particles::ParticleAdvector2D;

let advector = ParticleAdvector2D::new(stream.clone(), n, &mass_host)?;

// Each time step (after fluid.gather):
advector.compute_stokes_drag(&fluid.vel_x, &fluid.vel_y, mu, radius)?;
advector.update_velocity(dt)?;
advector.update_position(&mut px_d, &mut py_d, dt)?;
advector.apply_periodicity(&mut px_d, &mut py_d, x_lo, x_hi, y_lo, y_hi)?;
```

Both structs read kernels from `cuda_kernels/particles.ptx`.

---

### `filtered_mesh` — `StokesField`, `FilteredMesh`

```rust
use jaali_clow::filtered_mesh::{find_body_bounds, filter_mesh};

// Detect the obstacle bounding box from boundary edges
let (bx_lo, bx_hi, by_lo, by_hi) = find_body_bounds(&field);

// Clip the mesh to a spatial region and reindex vertices
let fmesh = filter_mesh(&field, x_lo, x_hi, y_lo, y_hi);
```

---

### `io` — VTK I/O and ParaView helpers

```rust
use jaali_clow::io::{
    read_vtk_2d,
    write_vtk_particles,
    write_vtk_submesh,
    write_visualize_py,
    write_reload_macro_py,
};

let field = read_vtk_2d("examples/acoustic_field.vtk");
write_vtk_particles("vtk_out/particles_000000.vtk", step, &px, &py, &u, &v, &p);
write_vtk_submesh("vtk_out/mesh.vtk", &fmesh);

// One-time setup: write ParaView scripts into the output directory
write_visualize_py("vtk_out", "acoustic_mesh_under_study.vtk", n_particles);
write_reload_macro_py("vtk_out");
```

`write_visualize_py` generates `vtk_out/visualize.py` — run with `pvpython` to open
the mesh and animate the particle snapshots in ParaView.

`write_reload_macro_py` generates `vtk_out/reload_macro.py` — register it once under
*Tools → Macros* in ParaView to reload all VTK readers from disk without restarting.

---

## Examples

### Tracer advection in Stokes flow

```bash
cargo run --example particles_in_stokes_2d -- --particles 200 --steps 500 --dt 0.01
```

Randomly seeds massless tracers and advects them with the local fluid velocity.

---

### Tracer advection in acoustic field

```bash
cargo run --example acoustic_particle_trapping_2d -- \
    --steps 200 --dt 0.05 --ppc 8
```

Places a uniform particle grid downstream of the obstacle and advects tracers
using the acoustic velocity field.

---

### Stokes-drag particle integrator

```bash
cargo run --example acoustic_particle_trapping_2d_01_only_stokes_drag -- \
    --steps 500 --dt 1e-4 --mu 1e-3 --radius 5e-6 --rho-p 1050
```

Integrates inertial spherical particles under Stokes drag:

```
F = -6π μ r (v_particle − v_fluid)
v_particle += dt · F / m
x_particle += dt · v_particle
```

#### Full option reference

| Flag | Default | Description |
|------|---------|-------------|
| `--steps N` | 100 | Number of time steps |
| `--out-every N` | 10 | Write VTK snapshot every N steps |
| `--dt F` | 0.1 | Time-step size |
| `--mu F` | 1e-3 | Fluid dynamic viscosity (Pa·s) |
| `--radius F` | 5e-6 | Particle radius (m) |
| `--rho-p F` | 1050 | Particle density (kg/m³) |
| `--ppc N` | 5 | Particles per body-width (sets grid spacing) |
| `--spacing F` | — | Override grid spacing directly |
| `--x-margin F` | 3.0 | Particle region width (× body width) |
| `--y-margin F` | 2.0 | Particle region height (× body height) |
| `--mesh-x-margin F` | 1.0 | Mesh margin in x (× body width) |
| `--mesh-y-margin F` | 1.0 | Mesh margin in y (× body height) |

---

## Tools

### `tools/find_period.py` — acoustic period detection

Reads a time-series of VTK field snapshots (`field-0.vtk`, `field-1.vtk`, …),
samples the pressure at a chosen triangle, detects oscillation peaks, and prints
the file list for the last complete acoustic cycle.  Useful as a preprocessing
step before computing time-averaged acoustic radiation forces.

#### Usage

```bash
# Print the cycle file list
python tools/find_period.py /path/to/vtk_series/

# Copy last-cycle files into a new folder
python tools/find_period.py /path/to/vtk_series/ --out-dir /tmp/one_period/

# Show a pressure plot with detected peaks highlighted
python tools/find_period.py /path/to/vtk_series/ --plot

# Sample a different triangle (useful if triangle 0 is in a low-pressure region)
python tools/find_period.py /path/to/vtk_series/ --probe-tri 500
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `folder` | (required) | Directory containing `field-N.vtk` files |
| `--out-dir DIR` | — | Copy selected files into DIR |
| `--probe-tri N` | 0 | Triangle index to sample pressure from |
| `--min-prominence F` | 0.2 | Peak height threshold as fraction of signal range |
| `--plot` | off | Show matplotlib plot of pressure signal and peaks |

#### Algorithm

1. Glob all `*-<N>.vtk` files, sort numerically by `N`.
2. Read `pressure[probe_tri]` from each file (legacy ASCII VTK parser).
3. Detect local maxima above the prominence threshold.
4. Average consecutive peak gaps to estimate the cycle length.
5. Slice `files[peaks[-2] : peaks[-1]]` — the last stable full cycle.

> **No `dt` required.** The period is measured directly in file-index units from
> the pressure signal, so the script works regardless of the time-step size used
> to generate the snapshots.

---

## Output files

All examples write into `vtk_out/`:

| File | Description |
|------|-------------|
| `acoustic_mesh_under_study.vtk` | Filtered sub-mesh used for the simulation |
| `particles_NNNNNN.vtk` | Particle snapshot at step NNNNNN (POLYDATA) |
| `visualize.py` | ParaView script — run with `pvpython visualize.py` |
| `reload_macro.py` | ParaView macro — reload readers without restarting |


# Analysis

```bash
dineshadepu@dwi199a (main) /home/dineshadepu/life/softwares/jaali_clow $
|  lab desktop => python tools/find_period.py /home/dineshadepu/postdoc_dwi/softwares/AcoDyn/sim1/time-series
found 406 VTK files  [0 … 405]
reading pressure at triangle 0 from each file …
pressure range: [-9.9935e+04, 9.9932e+04]
peaks at file-indices: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 259, 263, 267, 271, 275, 279, 283, 287, 291, 295, 299, 303, 307, 311, 315, 319, 323, 327, 331, 335, 339, 343, 347, 351, 355, 359, 363, 367, 371, 375, 379, 383, 387, 391, 395, 399, 403]
cycle length: 4.00 frames  (min=4, max=4, n_cycles=100)

last cycle: 4 files  [field-399.vtk … field-402.vtk]
  /home/dineshadepu/postdoc_dwi/softwares/AcoDyn/sim1/time-series/field-399.vtk
  /home/dineshadepu/postdoc_dwi/softwares/AcoDyn/sim1/time-series/field-400.vtk
  /home/dineshadepu/postdoc_dwi/softwares/AcoDyn/sim1/time-series/field-401.vtk
  /home/dineshadepu/postdoc_dwi/softwares/AcoDyn/sim1/time-series/field-402.vtk
```
