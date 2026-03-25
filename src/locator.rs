use crate::bvh::{Bvh2DGPU};
use crate::mesh::{TriMeshGPU};

use crate::gpu::*;

use std::sync::Arc;

const MAX_HITS_2D: usize = 32;

/* ========================== 2D ========================== */
struct Locator2D{
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    reduce_kernel: CudaFunction,

    bvh: Bvh2DGPU,
    mesh: TriMeshGPU,

    max_hits: usize,
    indices: CudaSlice<i32>,
    counts: CudaSlice<u16>,
}

impl Locator2D {
    /// Minimal constructor (single-hit legacy usage)
    pub fn new(mesh: &TriMesh<'_>) -> GpuResult<Self> {
        Self::new_with_capacity(mesh, 0, MAX_HITS_2D)
    }

    /// Authoritative constructor
    pub fn new_with_capacity(mesh: &TriMesh<'_>, max_queries: usize, max_hits: usize) -> GpuResult<Self> {

        let cuda = CudaManager::new(0)?;
        let stream = cuda.new_stream()?;
        // clone for later use
        let stream_for_buffers = stream.clone();

        let bvh = crate::bvh::Bvh2D::build(mesh);
        let mesh_gpu = mesh.to_gpu(stream.clone())?;
        let bvh_gpu = bvh.to_gpu(stream.clone())?;

        let module = cuda.load_module("cuda_kernels/locate_triangles.ptx")?;
        let kernel = module.get("locate_triangles_all")?;
        let reduce_kernel = module.get("select_min_owner")?;

        Ok(Locator2D{
            stream,
            kernel,
            reduce_kernel,
            bvh: bvh_gpu,
            mesh: mesh_gpu,
            max_hits: max_hits,
            indices: stream_for_buffers.clone().alloc_zeros::<i32>(0)?,
            counts: stream_for_buffers.clone().alloc_zeros::<u16>(0)?,
        })
    }

    fn locate_all(&mut self, qx_d: &CudaSlice<f64>, qy_d: &CudaSlice<f64>) -> GpuResult<()> {
        let n = qx_d.len();
        let H = self.max_hits;
        let required = n * H;

        if self.indices.len() < required {
            self.indices = self.stream.alloc_zeros::<i32>(required)?;
        }
        if self.counts.len() < n {
            self.counts = self.stream.alloc_zeros::<u16>(n)?;
        }

        // -------------------------------------------------
        // Clear counts on GPU (CRITICAL)
        // -------------------------------------------------
        self.stream.memset_zeros(&mut self.counts)?;

        // -------------------------------------------------
        // Launch kernel
        // -------------------------------------------------
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = self.stream.launch_builder(&self.kernel);

        launch.arg(qx_d);
        launch.arg(qy_d);
        launch.arg(&self.indices);
        launch.arg(&self.counts);

        let n_i32 = n as i32;
        let h_i32 = H as i32;
        launch.arg(&n_i32);
        launch.arg(&h_i32);

        // BVH
        launch.arg(&self.bvh.xmin);
        launch.arg(&self.bvh.ymin);
        launch.arg(&self.bvh.xmax);
        launch.arg(&self.bvh.ymax);
        launch.arg(&self.bvh.left);
        launch.arg(&self.bvh.right);
        launch.arg(&self.bvh.tri);

        // Mesh
        launch.arg(&self.mesh.vx);
        launch.arg(&self.mesh.vy);
        launch.arg(&self.mesh.t0);
        launch.arg(&self.mesh.t1);
        launch.arg(&self.mesh.t2);

        unsafe { launch.launch(cfg)? };

        // // -------------------------------------------------
        // // Copy results back (only if CPU needs them)
        // // -------------------------------------------------
        // gpu.stream.memcpy_dtoh(&gpu.indices, &mut self.indices)?;
        // gpu.stream.memcpy_dtoh(&gpu.counts, &mut self.counts)?;

        Ok(())
    }

    pub fn locate(&mut self, qx_d: &CudaSlice<f64>, qy_d: &CudaSlice<f64>, out: &mut CudaSlice<i32>) -> GpuResult<()> {
        let n = qx_d.len();
        assert_eq!(out.len(), n);

        self.locate_all(qx_d, qy_d)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = self.stream.launch_builder(&self.reduce_kernel);

        launch.arg(&self.indices);
        launch.arg(&self.counts);
        launch.arg(out);

        let n_i32 = n as i32;
        let h_i32 = self.max_hits as i32;
        launch.arg(&n_i32);
        launch.arg(&h_i32);

        unsafe { launch.launch(cfg)? };

        Ok(())
    }
}



use crate::mesh::{TriMesh};

pub fn single_triangle() -> TriMesh<'static> {
    let vx = Box::leak(vec![0.0, 1.0, 0.0].into_boxed_slice());
    let vy = Box::leak(vec![0.0, 0.0, 1.0].into_boxed_slice());

    let t0 = Box::leak(vec![0usize].into_boxed_slice());
    let t1 = Box::leak(vec![1usize].into_boxed_slice());
    let t2 = Box::leak(vec![2usize].into_boxed_slice());

    TriMesh { vx, vy, t0, t1, t2 }
}

#[test]
fn locator2d_basic_inside_all_backends() {
    // Single triangle
    let mesh = single_triangle();

    // Strictly inside
    let qx = vec![0.25];
    let qy = vec![0.25];

    let mut locator = Locator2D::new(&mesh).unwrap();

    let qx_d = locator.stream.clone_htod(&qx).unwrap();
    let qy_d = locator.stream.clone_htod(&qy).unwrap();
    let mut out_d = locator.stream.alloc_zeros::<i32>(qx.len()).unwrap();
    locator.locate(&qx_d, &qy_d, &mut out_d).unwrap();

    let mut out = vec![-1; qx.len()];
    locator.stream.memcpy_dtoh(&out_d, &mut out).unwrap();

    assert_eq!(out[0], 0, "Locator2D failed");
}
