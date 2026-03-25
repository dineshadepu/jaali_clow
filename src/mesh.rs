use crate::gpu::*;
use std::sync::Arc;

pub struct TriMesh<'a> {
    pub vx: &'a [f64],
    pub vy: &'a [f64],
    pub t0: &'a [usize],
    pub t1: &'a [usize],
    pub t2: &'a [usize],
}

pub struct TetMesh<'a> {
    pub vx: &'a [f64],
    pub vy: &'a [f64],
    pub vz: &'a [f64],
    pub t0: &'a [usize],
    pub t1: &'a [usize],
    pub t2: &'a [usize],
    pub t3: &'a [usize],
}

pub struct TetMeshGPU {
    pub vx: CudaSlice<f64>,
    pub vy: CudaSlice<f64>,
    pub vz: CudaSlice<f64>,
    pub t0: CudaSlice<i32>,
    pub t1: CudaSlice<i32>,
    pub t2: CudaSlice<i32>,
    pub t3: CudaSlice<i32>,
}

pub struct TriMeshGPU {
    pub vx: CudaSlice<f64>,
    pub vy: CudaSlice<f64>,
    pub t0: CudaSlice<i32>,
    pub t1: CudaSlice<i32>,
    pub t2: CudaSlice<i32>,
}

impl<'a> TriMesh<'a> {
    pub fn to_gpu(&self, stream: Arc<CudaStream>) -> GpuResult<TriMeshGPU> {
        // --- convert connectivity to i32 on host ---
        let t0_i32: Vec<i32> = self.t0.iter().map(|&x| x as i32).collect();
        let t1_i32: Vec<i32> = self.t1.iter().map(|&x| x as i32).collect();
        let t2_i32: Vec<i32> = self.t2.iter().map(|&x| x as i32).collect();

        Ok(TriMeshGPU {
            vx: stream.clone_htod(self.vx)?,
            vy: stream.clone_htod(self.vy)?,
            t0: stream.clone_htod(&t0_i32)?,
            t1: stream.clone_htod(&t1_i32)?,
            t2: stream.clone_htod(&t2_i32)?,
        })
    }
}

impl<'a> TetMesh<'a> {
    pub fn to_gpu(&self, stream: Arc<CudaStream>) -> GpuResult<TetMeshGPU> {
        // --- convert connectivity to i32 on host ---
        let t0_i32: Vec<i32> = self.t0.iter().map(|&x| x as i32).collect();
        let t1_i32: Vec<i32> = self.t1.iter().map(|&x| x as i32).collect();
        let t2_i32: Vec<i32> = self.t2.iter().map(|&x| x as i32).collect();
        let t3_i32: Vec<i32> = self.t3.iter().map(|&x| x as i32).collect();

        Ok(TetMeshGPU {
            vx: stream.clone_htod(self.vx)?,
            vy: stream.clone_htod(self.vy)?,
            vz: stream.clone_htod(self.vz)?,
            t0: stream.clone_htod(&t0_i32)?,
            t1: stream.clone_htod(&t1_i32)?,
            t2: stream.clone_htod(&t2_i32)?,
            t3: stream.clone_htod(&t3_i32)?,
        })
    }
}
