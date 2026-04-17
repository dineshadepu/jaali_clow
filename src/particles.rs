use crate::gpu::*;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// FluidField2D
// Holds cell-centred fluid data uploaded from the mesh and gathers it onto
// per-particle arrays after each locator step.
// ---------------------------------------------------------------------------
pub struct FluidField2D {
    pub stream: Arc<CudaStream>,
    gather_kernel: CudaFunction,
    advect_kernel: CudaFunction,
    wrap_kernel: CudaFunction,

    // per-cell arrays (GPU)
    cell_vel_x: CudaSlice<f64>,
    cell_vel_y: CudaSlice<f64>,
    cell_pressure: CudaSlice<f64>,

    // per-particle gathered values (GPU), grown lazily
    pub vel_x: CudaSlice<f64>,
    pub vel_y: CudaSlice<f64>,
    pub pressure: CudaSlice<f64>,
}

impl FluidField2D {
    pub fn new(
        stream: Arc<CudaStream>,
        vel_x_host: &[f64],
        vel_y_host: &[f64],
        pressure_host: &[f64],
    ) -> GpuResult<Self> {
        let cuda = CudaManager::new(0)?;
        let module = cuda.load_module("cuda_kernels/particles.ptx")?;

        let gather_kernel = module.get("gather_cell_data")?;
        let advect_kernel = module.get("advect_tracer_2d")?;
        let wrap_kernel = module.get("wrap_periodic_2d")?;

        let cell_vel_x = stream.clone_htod(vel_x_host)?;
        let cell_vel_y = stream.clone_htod(vel_y_host)?;
        let cell_pressure = stream.clone_htod(pressure_host)?;

        Ok(Self {
            stream: stream.clone(),
            gather_kernel,
            advect_kernel,
            wrap_kernel,
            cell_vel_x,
            cell_vel_y,
            cell_pressure,
            vel_x: stream.alloc_zeros::<f64>(0)?,
            vel_y: stream.alloc_zeros::<f64>(0)?,
            pressure: stream.alloc_zeros::<f64>(0)?,
        })
    }

    /// Gather cell-centred field values to per-particle arrays using located
    /// triangle IDs.
    pub fn gather(&mut self, triangle_ids: &CudaSlice<i32>) -> GpuResult<()> {
        let n = triangle_ids.len();

        if self.vel_x.len() < n {
            self.vel_x = self.stream.alloc_zeros::<f64>(n)?;
            self.vel_y = self.stream.alloc_zeros::<f64>(n)?;
            self.pressure = self.stream.alloc_zeros::<f64>(n)?;
        }

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = self.stream.launch_builder(&self.gather_kernel);
        launch.arg(triangle_ids);
        launch.arg(&self.cell_vel_x);
        launch.arg(&self.cell_vel_y);
        launch.arg(&self.cell_pressure);
        launch.arg(&self.vel_x);
        launch.arg(&self.vel_y);
        launch.arg(&self.pressure);
        let n_i32 = n as i32;
        launch.arg(&n_i32);
        unsafe { launch.launch(cfg)? };
        Ok(())
    }

    /// Tracer advection: pos += dt * gathered_fluid_vel.
    /// Use this when particles have no inertia and simply follow the flow.
    pub fn advect_tracer(
        &self,
        px: &mut CudaSlice<f64>,
        py: &mut CudaSlice<f64>,
        dt: f64,
    ) -> GpuResult<()> {
        let n = px.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = self.stream.launch_builder(&self.advect_kernel);
        launch.arg(px);
        launch.arg(py);
        launch.arg(&self.vel_x);
        launch.arg(&self.vel_y);
        launch.arg(&dt);
        let n_i32 = n as i32;
        launch.arg(&n_i32);
        unsafe { launch.launch(cfg)? };
        Ok(())
    }

    /// Wrap particle positions into the periodic domain.
    pub fn apply_periodicity(
        &self,
        px: &mut CudaSlice<f64>,
        py: &mut CudaSlice<f64>,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    ) -> GpuResult<()> {
        let n = px.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = self.stream.launch_builder(&self.wrap_kernel);
        launch.arg(px);
        launch.arg(py);
        launch.arg(&x_min);
        launch.arg(&x_max);
        launch.arg(&y_min);
        launch.arg(&y_max);
        let n_i32 = n as i32;
        launch.arg(&n_i32);
        unsafe { launch.launch(cfg)? };
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ParticleAdvector2D
// Force-based integrator: compute forces → update velocity → update position.
// Suitable for physics-accurate particles (e.g. acoustophoresis, Stokes drag).
// ---------------------------------------------------------------------------
pub struct ParticleAdvector2D {
    pub stream: Arc<CudaStream>,
    stokes_drag_kernel: CudaFunction,
    update_vel_kernel: CudaFunction,
    update_pos_kernel: CudaFunction,
    wrap_kernel: CudaFunction,

    pub vel_x_p: CudaSlice<f64>,
    pub vel_y_p: CudaSlice<f64>,
    pub force_x_p: CudaSlice<f64>,
    pub force_y_p: CudaSlice<f64>,
    pub mass_p: CudaSlice<f64>,
}

impl ParticleAdvector2D {
    /// Create advector for `n` particles.
    /// `mass_host` must have length `n`; particle velocities start at zero.
    pub fn new(
        stream: Arc<CudaStream>,
        n: usize,
        mass_host: &[f64],
    ) -> GpuResult<Self> {
        assert_eq!(mass_host.len(), n);

        let cuda = CudaManager::new(0)?;
        let module = cuda.load_module("cuda_kernels/particles.ptx")?;

        let stokes_drag_kernel = module.get("stokes_drag_2d")?;
        let update_vel_kernel = module.get("update_velocity_2d")?;
        let update_pos_kernel = module.get("update_position_2d")?;
        let wrap_kernel = module.get("wrap_periodic_2d")?;

        let mass_p = stream.clone_htod(mass_host)?;

        Ok(Self {
            stream: stream.clone(),
            stokes_drag_kernel,
            update_vel_kernel,
            update_pos_kernel,
            wrap_kernel,
            vel_x_p: stream.alloc_zeros::<f64>(n)?,
            vel_y_p: stream.alloc_zeros::<f64>(n)?,
            force_x_p: stream.alloc_zeros::<f64>(n)?,
            force_y_p: stream.alloc_zeros::<f64>(n)?,
            mass_p,
        })
    }

    /// Compute Stokes drag: F = -6π·mu·radius·(v_particle - v_fluid).
    /// Call this first in the integration loop.
    pub fn compute_stokes_drag(
        &self,
        fluid_vel_x: &CudaSlice<f64>,
        fluid_vel_y: &CudaSlice<f64>,
        mu: f64,
        radius: f64,
    ) -> GpuResult<()> {
        let n = self.vel_x_p.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = self.stream.launch_builder(&self.stokes_drag_kernel);
        launch.arg(fluid_vel_x);
        launch.arg(fluid_vel_y);
        launch.arg(&self.vel_x_p);
        launch.arg(&self.vel_y_p);
        launch.arg(&self.force_x_p);
        launch.arg(&self.force_y_p);
        launch.arg(&mu);
        launch.arg(&radius);
        let n_i32 = n as i32;
        launch.arg(&n_i32);
        unsafe { launch.launch(cfg)? };
        Ok(())
    }

    /// Update particle velocity: vel_p += dt * force_p / mass_p.
    /// Call after all force contributions have been accumulated.
    pub fn update_velocity(&self, dt: f64) -> GpuResult<()> {
        let n = self.vel_x_p.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = self.stream.launch_builder(&self.update_vel_kernel);
        launch.arg(&self.vel_x_p);
        launch.arg(&self.vel_y_p);
        launch.arg(&self.force_x_p);
        launch.arg(&self.force_y_p);
        launch.arg(&self.mass_p);
        launch.arg(&dt);
        let n_i32 = n as i32;
        launch.arg(&n_i32);
        unsafe { launch.launch(cfg)? };
        Ok(())
    }

    /// Update particle position: pos += dt * vel_p.
    /// Call after update_velocity.
    pub fn update_position(
        &self,
        px: &mut CudaSlice<f64>,
        py: &mut CudaSlice<f64>,
        dt: f64,
    ) -> GpuResult<()> {
        let n = px.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = self.stream.launch_builder(&self.update_pos_kernel);
        launch.arg(px);
        launch.arg(py);
        launch.arg(&self.vel_x_p);
        launch.arg(&self.vel_y_p);
        launch.arg(&dt);
        let n_i32 = n as i32;
        launch.arg(&n_i32);
        unsafe { launch.launch(cfg)? };
        Ok(())
    }

    /// Wrap particle positions into the periodic domain.
    pub fn apply_periodicity(
        &self,
        px: &mut CudaSlice<f64>,
        py: &mut CudaSlice<f64>,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    ) -> GpuResult<()> {
        let n = px.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = self.stream.launch_builder(&self.wrap_kernel);
        launch.arg(px);
        launch.arg(py);
        launch.arg(&x_min);
        launch.arg(&x_max);
        launch.arg(&y_min);
        launch.arg(&y_max);
        let n_i32 = n as i32;
        launch.arg(&n_i32);
        unsafe { launch.launch(cfg)? };
        Ok(())
    }
}
