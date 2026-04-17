use jaali_clow::io::{read_vtk_2d, write_reload_macro_py, write_visualize_py, write_vtk_particles};
use jaali_clow::locator::Locator2D;
use jaali_clow::mesh::TriMesh;
use jaali_clow::particles::FluidField2D;

// ------------------------------------------------------------
// CLI
// ------------------------------------------------------------

struct Args {
    steps:     usize,
    out_every: usize,
    dt:        f64,
    particles: usize,
    seed:      u64,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let mut steps     = 100usize;
    let mut out_every = 10usize;
    let mut dt        = 0.1f64;
    let mut particles = 4usize;
    let mut seed      = 42u64;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--steps"     => { steps     = raw[i+1].parse().unwrap(); i += 2; }
            "--out-every" => { out_every = raw[i+1].parse().unwrap(); i += 2; }
            "--dt"        => { dt        = raw[i+1].parse().unwrap(); i += 2; }
            "--particles" => { particles = raw[i+1].parse().unwrap(); i += 2; }
            "--seed"      => { seed      = raw[i+1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }
    Args { steps, out_every, dt, particles, seed }
}

// ------------------------------------------------------------
// Minimal xorshift64 PRNG
// ------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

fn main() {
    let args  = parse_args();
    let field = read_vtk_2d("examples/stokes_field_2d.vtk");
    println!("mesh: {} points, {} triangles", field.vx.len(), field.t0.len());

    let x_min = field.vx.iter().cloned().fold(f64::INFINITY,     f64::min);
    let x_max = field.vx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = field.vy.iter().cloned().fold(f64::INFINITY,     f64::min);
    let y_max = field.vy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("domain: x=[{:.4}, {:.4}]  y=[{:.4}, {:.4}]", x_min, x_max, y_min, y_max);

    let mesh = TriMesh {
        vx: &field.vx, vy: &field.vy,
        t0: &field.t0, t1: &field.t1, t2: &field.t2,
    };

    let mut locator     = Locator2D::new(&mesh).unwrap();
    let mut fluid_field = FluidField2D::new(
        locator.stream.clone(),
        &field.vel_x, &field.vel_y, &field.pressure,
    ).unwrap();

    let n = args.particles;
    let mut rng = Rng::new(args.seed);
    let qx_init: Vec<f64> = (0..n).map(|_| rng.uniform(x_min, x_max)).collect();
    let qy_init: Vec<f64> = (0..n).map(|_| rng.uniform(y_min, y_max)).collect();
    println!("particles={} seed={}", n, args.seed);

    let mut px_d  = locator.stream.clone_htod(&qx_init).unwrap();
    let mut py_d  = locator.stream.clone_htod(&qy_init).unwrap();
    let mut ids_d = locator.stream.alloc_zeros::<i32>(n).unwrap();

    let mut px_h  = vec![0f64; n];
    let mut py_h  = vec![0f64; n];
    let mut u_h   = vec![0f64; n];
    let mut v_h   = vec![0f64; n];
    let mut p_h   = vec![0f64; n];
    let mut ids_h = vec![-1i32; n];

    std::fs::create_dir_all("vtk_out").unwrap();
    write_visualize_py("vtk_out", "../examples/stokes_field_2d.vtk", n);
    write_reload_macro_py("vtk_out");

    println!("\nsteps={} out_every={} dt={}", args.steps, args.out_every, args.dt);

    for step in 0..=args.steps {
        locator.locate(&px_d, &py_d, &mut ids_d).unwrap();
        fluid_field.gather(&ids_d).unwrap();

        if step % args.out_every == 0 {
            locator.stream.memcpy_dtoh(&px_d,                &mut px_h).unwrap();
            locator.stream.memcpy_dtoh(&py_d,                &mut py_h).unwrap();
            locator.stream.memcpy_dtoh(&fluid_field.vel_x,   &mut u_h).unwrap();
            locator.stream.memcpy_dtoh(&fluid_field.vel_y,   &mut v_h).unwrap();
            locator.stream.memcpy_dtoh(&fluid_field.pressure, &mut p_h).unwrap();
            locator.stream.memcpy_dtoh(&ids_d,               &mut ids_h).unwrap();

            let vtk_path = format!("vtk_out/particles_{:06}.vtk", step);
            write_vtk_particles(&vtk_path, step, &px_h, &py_h, &u_h, &v_h, &p_h);
            println!("step {:>6}  px[0]={:.4}  py[0]={:.4}  tri[0]={}  p[0]={:.4}  → {}",
                     step, px_h[0], py_h[0], ids_h[0], p_h[0], vtk_path);
        }

        if step < args.steps {
            fluid_field.advect_tracer(&mut px_d, &mut py_d, args.dt).unwrap();
            fluid_field.apply_periodicity(
                &mut px_d, &mut py_d, x_min, x_max, y_min, y_max,
            ).unwrap();
        }
    }
}
