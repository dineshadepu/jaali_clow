use crate::mesh::{TetMesh, TriMesh};

use crate::gpu::*;
use std::sync::Arc;

#[inline(always)]
fn triangle_aabb(i: usize, mesh: &TriMesh) -> (f64, f64, f64, f64) {
    let i0 = mesh.t0[i];
    let i1 = mesh.t1[i];
    let i2 = mesh.t2[i];

    let x0 = mesh.vx[i0];
    let y0 = mesh.vy[i0];
    let x1 = mesh.vx[i1];
    let y1 = mesh.vy[i1];
    let x2 = mesh.vx[i2];
    let y2 = mesh.vy[i2];

    (
        x0.min(x1.min(x2)),
        y0.min(y1.min(y2)),
        x0.max(x1.max(x2)),
        y0.max(y1.max(y2)),
    )
}

pub struct Bvh3DGPU {
    pub xmin: CudaSlice<f64>,
    pub ymin: CudaSlice<f64>,
    pub zmin: CudaSlice<f64>,
    pub xmax: CudaSlice<f64>,
    pub ymax: CudaSlice<f64>,
    pub zmax: CudaSlice<f64>,
    pub left: CudaSlice<i32>,
    pub right: CudaSlice<i32>,
    pub tet: CudaSlice<i32>,
}

pub struct Bvh2DGPU {
    pub xmin: CudaSlice<f64>,
    pub ymin: CudaSlice<f64>,
    pub xmax: CudaSlice<f64>,
    pub ymax: CudaSlice<f64>,
    pub left: CudaSlice<i32>,
    pub right: CudaSlice<i32>,
    pub tri: CudaSlice<i32>,
}

pub struct Bvh2D {
    pub xmin: Vec<f64>,
    pub ymin: Vec<f64>,
    pub xmax: Vec<f64>,
    pub ymax: Vec<f64>,
    pub left: Vec<i32>,
    pub right: Vec<i32>,
    pub tri: Vec<i32>,
}

impl Bvh2D {
    pub fn build(mesh: &TriMesh) -> Self {
        let mut bvh = Self {
            xmin: Vec::new(),
            ymin: Vec::new(),
            xmax: Vec::new(),
            ymax: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            tri: Vec::new(),
        };

        let mut tris: Vec<usize> = (0..mesh.t0.len()).collect();
        Self::build_node(&mut tris, mesh, &mut bvh);
        bvh
    }

    fn build_node(tris: &mut [usize], mesh: &TriMesh, bvh: &mut Bvh2D) -> i32 {
        let node = bvh.xmin.len() as i32;

        let (mut xmin, mut ymin, mut xmax, mut ymax) = triangle_aabb(tris[0], mesh);

        for &t in &tris[1..] {
            let (x0, y0, x1, y1) = triangle_aabb(t, mesh);
            xmin = xmin.min(x0);
            ymin = ymin.min(y0);
            xmax = xmax.max(x1);
            ymax = ymax.max(y1);
        }

        bvh.xmin.push(xmin);
        bvh.ymin.push(ymin);
        bvh.xmax.push(xmax);
        bvh.ymax.push(ymax);
        bvh.left.push(-1);
        bvh.right.push(-1);
        bvh.tri.push(-1);

        if tris.len() == 1 {
            bvh.tri[node as usize] = tris[0] as i32;
            return node;
        }

        let axis = if xmax - xmin > ymax - ymin { 0 } else { 1 };

        tris.sort_by(|&a, &b| {
            let (ax0, ay0, ax1, ay1) = triangle_aabb(a, mesh);
            let (bx0, by0, bx1, by1) = triangle_aabb(b, mesh);
            let ca = if axis == 0 { ax0 + ax1 } else { ay0 + ay1 };
            let cb = if axis == 0 { bx0 + bx1 } else { by0 + by1 };
            ca.partial_cmp(&cb).unwrap()
        });

        let mid = tris.len() / 2;
        let (l, r) = tris.split_at_mut(mid);

        let left = Self::build_node(l, mesh, bvh);
        let right = Self::build_node(r, mesh, bvh);

        bvh.left[node as usize] = left;
        bvh.right[node as usize] = right;

        node
    }

    pub fn to_gpu(&self, stream: Arc<CudaStream>) -> GpuResult<Bvh2DGPU> {
        debug_assert_eq!(self.xmin.len(), self.xmax.len());
        debug_assert_eq!(self.xmin.len(), self.ymin.len());
        debug_assert_eq!(self.xmin.len(), self.left.len());
        debug_assert_eq!(self.xmin.len(), self.right.len());
        debug_assert_eq!(self.xmin.len(), self.tri.len());

        Ok(Bvh2DGPU {
            xmin: stream.clone_htod(&self.xmin)?,
            ymin: stream.clone_htod(&self.ymin)?,
            xmax: stream.clone_htod(&self.xmax)?,
            ymax: stream.clone_htod(&self.ymax)?,
            left: stream.clone_htod(&self.left)?,
            right: stream.clone_htod(&self.right)?,
            tri: stream.clone_htod(&self.tri)?,
        })
    }
}

#[inline(always)]
fn tet_aabb(i: usize, mesh: &TetMesh) -> (f64, f64, f64, f64, f64, f64) {
    let ids = [mesh.t0[i], mesh.t1[i], mesh.t2[i], mesh.t3[i]];

    let mut xmin = mesh.vx[ids[0]];
    let mut ymin = mesh.vy[ids[0]];
    let mut zmin = mesh.vz[ids[0]];
    let mut xmax = xmin;
    let mut ymax = ymin;
    let mut zmax = zmin;

    for &id in &ids[1..] {
        let x = mesh.vx[id];
        let y = mesh.vy[id];
        let z = mesh.vz[id];

        xmin = xmin.min(x);
        ymin = ymin.min(y);
        zmin = zmin.min(z);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
        zmax = zmax.max(z);
    }

    (xmin, ymin, zmin, xmax, ymax, zmax)
}

pub struct Bvh3D {
    pub xmin: Vec<f64>,
    pub ymin: Vec<f64>,
    pub zmin: Vec<f64>,
    pub xmax: Vec<f64>,
    pub ymax: Vec<f64>,
    pub zmax: Vec<f64>,
    pub left: Vec<i32>,
    pub right: Vec<i32>,
    pub tet: Vec<i32>,
}

impl Bvh3D {
    pub fn build(mesh: &TetMesh) -> Self {
        let mut bvh = Self {
            xmin: Vec::new(),
            ymin: Vec::new(),
            zmin: Vec::new(),
            xmax: Vec::new(),
            ymax: Vec::new(),
            zmax: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            tet: Vec::new(),
        };

        let mut tets: Vec<usize> = (0..mesh.t0.len()).collect();
        Self::build_node(&mut tets, mesh, &mut bvh);
        bvh
    }

    fn build_node(tets: &mut [usize], mesh: &TetMesh, bvh: &mut Bvh3D) -> i32 {
        let node = bvh.xmin.len() as i32;

        let (mut xmin, mut ymin, mut zmin, mut xmax, mut ymax, mut zmax) = tet_aabb(tets[0], mesh);

        for &t in &tets[1..] {
            let (x0, y0, z0, x1, y1, z1) = tet_aabb(t, mesh);
            xmin = xmin.min(x0);
            ymin = ymin.min(y0);
            zmin = zmin.min(z0);
            xmax = xmax.max(x1);
            ymax = ymax.max(y1);
            zmax = zmax.max(z1);
        }

        bvh.xmin.push(xmin);
        bvh.ymin.push(ymin);
        bvh.zmin.push(zmin);
        bvh.xmax.push(xmax);
        bvh.ymax.push(ymax);
        bvh.zmax.push(zmax);
        bvh.left.push(-1);
        bvh.right.push(-1);
        bvh.tet.push(-1);

        // Leaf
        if tets.len() == 1 {
            bvh.tet[node as usize] = tets[0] as i32;
            return node;
        }

        // Choose split axis
        let dx = xmax - xmin;
        let dy = ymax - ymin;
        let dz = zmax - zmin;
        let axis = if dx > dy && dx > dz {
            0
        } else if dy > dz {
            1
        } else {
            2
        };

        tets.sort_by(|&a, &b| {
            let ca = tet_aabb(a, mesh);
            let cb = tet_aabb(b, mesh);

            let ma = match axis {
                0 => ca.0 + ca.3,
                1 => ca.1 + ca.4,
                _ => ca.2 + ca.5,
            };
            let mb = match axis {
                0 => cb.0 + cb.3,
                1 => cb.1 + cb.4,
                _ => cb.2 + cb.5,
            };

            ma.partial_cmp(&mb).unwrap()
        });

        let mid = tets.len() / 2;
        let (l, r) = tets.split_at_mut(mid);

        let left = Self::build_node(l, mesh, bvh);
        let right = Self::build_node(r, mesh, bvh);

        bvh.left[node as usize] = left;
        bvh.right[node as usize] = right;

        node
    }

    pub fn to_gpu(&self, stream: Arc<CudaStream>) -> GpuResult<Bvh3DGPU> {
        debug_assert_eq!(self.xmin.len(), self.xmax.len());
        debug_assert_eq!(self.xmin.len(), self.ymin.len());
        debug_assert_eq!(self.xmin.len(), self.zmin.len());
        debug_assert_eq!(self.xmin.len(), self.left.len());
        debug_assert_eq!(self.xmin.len(), self.right.len());
        debug_assert_eq!(self.xmin.len(), self.tet.len());

        Ok(Bvh3DGPU {
            xmin: stream.clone_htod(&self.xmin)?,
            ymin: stream.clone_htod(&self.ymin)?,
            zmin: stream.clone_htod(&self.zmin)?,
            xmax: stream.clone_htod(&self.xmax)?,
            ymax: stream.clone_htod(&self.ymax)?,
            zmax: stream.clone_htod(&self.zmax)?,
            left: stream.clone_htod(&self.left)?,
            right: stream.clone_htod(&self.right)?,
            tet: stream.clone_htod(&self.tet)?,
        })
    }
}
