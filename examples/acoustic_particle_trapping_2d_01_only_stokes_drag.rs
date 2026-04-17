use jaali_clow::filtered_mesh::{filter_mesh, find_body_bounds};
use jaali_clow::io::{read_vtk_2d, write_reload_macro_py, write_visualize_py, write_vtk_particles, write_vtk_submesh};
use jaali_clow::locator::Locator2D;
use jaali_clow::mesh::TriMesh;
use jaali_clow::particles::{FluidField2D, ParticleAdvector2D};

// ------------------------------------------------------------
// CLI
// ------------------------------------------------------------

struct Args {
    steps: usize,
    out_every: usize,
    dt: f64,
    ppc: usize,
    spacing: Option<f64>,
    x_margin: f64,
    y_margin: f64,
    mesh_x_margin: f64,
    mesh_y_margin: f64,
    /// Dynamic viscosity of the fluid (Pa·s).
    mu: f64,
    /// Particle radius (m).
    radius: f64,
    /// Particle density (kg/m³), used to compute mass = rho_p * (4/3)π r³.
    rho_p: f64,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let mut steps         = 100usize;
    let mut out_every     = 10usize;
    let mut dt            = 0.1f64;
    let mut ppc           = 5usize;
    let mut spacing       = None::<f64>;
    let mut x_margin      = 3.0f64;
    let mut y_margin      = 2.0f64;
    let mut mesh_x_margin = 1.0f64;
    let mut mesh_y_margin = 1.0f64;
    let mut mu            = 1e-3f64;
    let mut radius        = 5e-6f64;
    let mut rho_p         = 1050.0f64;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--help" | "-h" => {
                println!("\
acoustic_particle_trapping_2d_01_only_stokes_drag — Stokes-drag particle integrator

USAGE:
  cargo run --example acoustic_particle_trapping_2d_01_only_stokes_drag -- [OPTIONS]

SIMULATION
  --steps N          Number of time steps to run            (default: 100)
  --out-every N      Write a VTK snapshot every N steps     (default: 10)
  --dt F             Time step size                         (default: 0.1)

PARTICLE PHYSICS
  --mu F             Fluid dynamic viscosity (Pa·s)         (default: 1e-3)
  --radius F         Particle radius (m)                    (default: 5e-6)
  --rho-p F          Particle density (kg/m³)               (default: 1050.0)

PARTICLE GRID  (placed in front of the body, upstream)
  --ppc N            Particles per body-width in each direction (default: 5)
  --spacing F        Grid spacing in mesh units (overrides --ppc)
  --x-margin F       Particle region width (multiples of body_width, default: 3.0)
  --y-margin F       Particle region height (multiples of body_height, default: 2.0)

MESH
  --mesh-x-margin F  Extra mesh margin in x (multiples of body_width, default: 1.0)
  --mesh-y-margin F  Extra mesh margin in y (multiples of body_height, default: 1.0)
");
                std::process::exit(0);
            }
            "--steps"         => { steps         = raw[i+1].parse().unwrap(); i += 2; }
            "--out-every"     => { out_every      = raw[i+1].parse().unwrap(); i += 2; }
            "--dt"            => { dt             = raw[i+1].parse().unwrap(); i += 2; }
            "--ppc"           => { ppc            = raw[i+1].parse().unwrap(); i += 2; }
            "--spacing"       => { spacing        = Some(raw[i+1].parse().unwrap()); i += 2; }
            "--x-margin"      => { x_margin       = raw[i+1].parse().unwrap(); i += 2; }
            "--y-margin"      => { y_margin       = raw[i+1].parse().unwrap(); i += 2; }
            "--mesh-x-margin" => { mesh_x_margin  = raw[i+1].parse().unwrap(); i += 2; }
            "--mesh-y-margin" => { mesh_y_margin  = raw[i+1].parse().unwrap(); i += 2; }
            "--mu"            => { mu             = raw[i+1].parse().unwrap(); i += 2; }
            "--radius"        => { radius         = raw[i+1].parse().unwrap(); i += 2; }
            "--rho-p"         => { rho_p          = raw[i+1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown option: {other}  (try --help)"); i += 1; }
        }
    }
    Args { steps, out_every, dt, ppc, spacing, x_margin, y_margin,
           mesh_x_margin, mesh_y_margin, mu, radius, rho_p }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

fn main() {
    let args  = parse_args();
    let field = read_vtk_2d("examples/acoustic_field.vtk");
    println!("mesh: {} points, {} triangles", field.vx.len(), field.t0.len());

    // Detect body bounding box from internal boundary edges.
    print!("detecting body bounds... ");
    let (bx_lo, bx_hi, by_lo, by_hi) = find_body_bounds(&field);
    let body_width  = bx_hi - bx_lo;
    let body_height = by_hi - by_lo;
    println!("body x=[{:.1}, {:.1}]  y=[{:.1}, {:.1}]  width={:.1}",
             bx_lo, bx_hi, by_lo, by_hi, body_width);

    // Particle region: in front of the body.
    let p_x_lo = bx_hi;
    let p_x_hi = bx_hi + args.x_margin * body_width;
    let p_y_lo = by_lo - args.y_margin * body_height;
    let p_y_hi = by_hi + args.y_margin * body_height;
    println!("particle region: x=[{:.1}, {:.1}]  y=[{:.1}, {:.1}]",
             p_x_lo, p_x_hi, p_y_lo, p_y_hi);

    // Mesh region: particle extent + margin on all sides.
    let x_lo = (bx_lo - args.mesh_x_margin * body_width).min(bx_lo - body_width);
    let x_hi = p_x_hi + args.mesh_x_margin * body_width;
    let y_lo = p_y_lo - args.mesh_y_margin * body_height;
    let y_hi = p_y_hi + args.mesh_y_margin * body_height;
    println!("mesh region:     x=[{:.1}, {:.1}]  y=[{:.1}, {:.1}]",
             x_lo, x_hi, y_lo, y_hi);

    std::fs::create_dir_all("vtk_out").unwrap();

    let fmesh     = filter_mesh(&field, x_lo, x_hi, y_lo, y_hi);
    let sub_y_min = fmesh.vy.iter().cloned().fold(f64::INFINITY,     f64::min);
    let sub_y_max = fmesh.vy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("filtered mesh: {} points, {} triangles  x=[{:.1},{:.1}] y=[{:.1},{:.1}]",
             fmesh.vx.len(), fmesh.t0.len(), x_lo, x_hi, sub_y_min, sub_y_max);

    write_vtk_submesh("vtk_out/acoustic_mesh_under_study.vtk", &fmesh);

    // Use the full mesh for the locator so particles always find a valid cell.
    let full_mesh = TriMesh {
        vx: &field.vx, vy: &field.vy,
        t0: &field.t0, t1: &field.t1, t2: &field.t2,
    };

    let mut locator     = Locator2D::new(&full_mesh).unwrap();
    let mut fluid_field = FluidField2D::new(
        locator.stream.clone(),
        &field.vel_x, &field.vel_y, &field.pressure,
    ).unwrap();

    // Build uniform particle grid in front of the body.
    let spacing = args.spacing.unwrap_or(body_width / args.ppc as f64);
    let p_x_lo  = p_x_lo.max(x_lo);
    let p_x_hi  = p_x_hi.min(x_hi);
    let p_y_lo  = p_y_lo.max(sub_y_min);
    let p_y_hi  = p_y_hi.min(sub_y_max);
    let nx      = ((p_x_hi - p_x_lo) / spacing).ceil() as usize;
    let ny      = ((p_y_hi - p_y_lo) / spacing).ceil() as usize;
    let n       = nx * ny;
    println!("particle grid: {}×{}={} (spacing={:.2})  x=[{:.1},{:.1}] y=[{:.1},{:.1}]",
             nx, ny, n, spacing, p_x_lo, p_x_hi, p_y_lo, p_y_hi);

    let mut qx_init = Vec::with_capacity(n);
    let mut qy_init = Vec::with_capacity(n);
    for ix in 0..nx {
        let x = p_x_lo + (ix as f64 + 0.5) * spacing;
        for iy in 0..ny {
            qx_init.push(x);
            qy_init.push(p_y_lo + (iy as f64 + 0.5) * spacing);
        }
    }

    // Particle mass: m = rho_p * (4/3) * pi * r^3
    let mass      = args.rho_p * (4.0 / 3.0) * std::f64::consts::PI * args.radius.powi(3);
    let mass_host = vec![mass; n];
    println!("physics: mu={:.2e}  radius={:.2e}  rho_p={:.1}  mass={:.4e}",
             args.mu, args.radius, args.rho_p, mass);

    let advector = ParticleAdvector2D::new(locator.stream.clone(), n, &mass_host).unwrap();

    let mut px_d  = locator.stream.clone_htod(&qx_init).unwrap();
    let mut py_d  = locator.stream.clone_htod(&qy_init).unwrap();
    let mut ids_d = locator.stream.alloc_zeros::<i32>(n).unwrap();

    let mut px_h  = vec![0f64; n];
    let mut py_h  = vec![0f64; n];
    let mut u_h   = vec![0f64; n];
    let mut v_h   = vec![0f64; n];
    let mut p_h   = vec![0f64; n];
    let mut ids_h = vec![-1i32; n];

    write_visualize_py("vtk_out", "acoustic_mesh_under_study.vtk", n);
    write_reload_macro_py("vtk_out");

    println!("\nsteps={} out_every={} dt={}", args.steps, args.out_every, args.dt);

    for step in 0..=args.steps {
        // 1. Locate particles and gather fluid field values.
        locator.locate(&px_d, &py_d, &mut ids_d).unwrap();
        fluid_field.gather(&ids_d).unwrap();

        if step % args.out_every == 0 {
            locator.stream.memcpy_dtoh(&px_d,              &mut px_h).unwrap();
            locator.stream.memcpy_dtoh(&py_d,              &mut py_h).unwrap();
            locator.stream.memcpy_dtoh(&fluid_field.vel_x, &mut u_h).unwrap();
            locator.stream.memcpy_dtoh(&fluid_field.vel_y, &mut v_h).unwrap();
            locator.stream.memcpy_dtoh(&fluid_field.pressure, &mut p_h).unwrap();
            locator.stream.memcpy_dtoh(&ids_d,             &mut ids_h).unwrap();

            let vtk_path = format!("vtk_out/particles_{:06}.vtk", step);
            write_vtk_particles(&vtk_path, step, &px_h, &py_h, &u_h, &v_h, &p_h);
            println!("step {:>6}  px[0]={:.4}  py[0]={:.4}  tri[0]={}  p[0]={:.4}  → {}",
                     step, px_h[0], py_h[0], ids_h[0], p_h[0], vtk_path);
        }

        if step < args.steps {
            // 2. Compute forces.
            advector.compute_stokes_drag(
                &fluid_field.vel_x, &fluid_field.vel_y, args.mu, args.radius,
            ).unwrap();

            // 3. Update velocity: vel_p += dt * force / mass.
            advector.update_velocity(args.dt).unwrap();

            // 4. Update position: pos += dt * vel_p.
            advector.update_position(&mut px_d, &mut py_d, args.dt).unwrap();

            // 5. Wrap into periodic domain.
            advector.apply_periodicity(
                &mut px_d, &mut py_d, x_lo, x_hi, sub_y_min, sub_y_max,
            ).unwrap();
        }
    }
}
