// src/gpu/cuda.rs
use cudarc::driver::*;
pub use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

pub type GpuResult<T> = Result<T, DriverError>;

pub struct CudaManager {
    pub ctx: Arc<CudaContext>,
}

impl CudaManager {
    pub fn new(device: usize) -> GpuResult<Self> {
        let ctx = CudaContext::new(device)?;
        Ok(Self { ctx })
    }

    pub fn new_stream(&self) -> GpuResult<Arc<CudaStream>> {
        Ok(self.ctx.new_stream()?)
    }

    pub fn load_module(&self, path: &str) -> GpuResult<ModuleHandle> {
        let ptx = std::fs::read_to_string(path).expect("Failed to read PTX file");
        let module = self.ctx.load_module(ptx.into())?;
        Ok(ModuleHandle { module })
    }
}

pub struct ModuleHandle {
    pub module: Arc<CudaModule>,
}

impl ModuleHandle {
    pub fn get(&self, name: &str) -> GpuResult<CudaFunction> {
        Ok(self.module.load_function(name)?)
    }
}
