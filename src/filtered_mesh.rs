use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// StokesField
// A triangle mesh together with per-cell velocity and pressure fields,
// as read from a VTK file.
// ---------------------------------------------------------------------------
pub struct StokesField {
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub t0: Vec<usize>,
    pub t1: Vec<usize>,
    pub t2: Vec<usize>,
    pub vel_x: Vec<f64>,
    pub vel_y: Vec<f64>,
    pub pressure: Vec<f64>,
}

// ---------------------------------------------------------------------------
// FilteredMesh
// A spatially clipped subset of a StokesField with reindexed vertices.
// ---------------------------------------------------------------------------
pub struct FilteredMesh {
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub t0: Vec<usize>,
    pub t1: Vec<usize>,
    pub t2: Vec<usize>,
    pub vel_x: Vec<f64>,
    pub vel_y: Vec<f64>,
    pub pressure: Vec<f64>,
}

/// Detect the bounding box of the immersed body by finding boundary edges
/// (edges shared by exactly one triangle) whose vertices are NOT on the outer
/// domain boundary.
///
/// Returns `(x_min, x_max, y_min, y_max)` of the body.
pub fn find_body_bounds(field: &StokesField) -> (f64, f64, f64, f64) {
    let dom_x_min = field.vx.iter().cloned().fold(f64::INFINITY,     f64::min);
    let dom_x_max = field.vx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let dom_y_min = field.vy.iter().cloned().fold(f64::INFINITY,     f64::min);
    let dom_y_max = field.vy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tol = 5.0;

    let mut edge_count: HashMap<(usize, usize), u32> = HashMap::new();
    for k in 0..field.t0.len() {
        for &(a, b) in &[
            (field.t0[k], field.t1[k]),
            (field.t1[k], field.t2[k]),
            (field.t2[k], field.t0[k]),
        ] {
            *edge_count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }

    let mut bx_min = f64::INFINITY;
    let mut bx_max = f64::NEG_INFINITY;
    let mut by_min = f64::INFINITY;
    let mut by_max = f64::NEG_INFINITY;

    for (&(a, b), &cnt) in &edge_count {
        if cnt != 1 {
            continue;
        }
        for &node in &[a, b] {
            let x = field.vx[node];
            let y = field.vy[node];
            if y < dom_y_min + tol || y > dom_y_max - tol
            || x < dom_x_min + tol || x > dom_x_max - tol
            {
                continue;
            }
            bx_min = bx_min.min(x);
            bx_max = bx_max.max(x);
            by_min = by_min.min(y);
            by_max = by_max.max(y);
        }
    }

    (bx_min, bx_max, by_min, by_max)
}

/// Extract triangles whose vertices are all within [x_lo, x_hi] × [y_lo, y_hi]
/// and reindex the vertices.
pub fn filter_mesh(
    field: &StokesField,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
) -> FilteredMesh {
    let inside = |n: usize| {
        field.vx[n] >= x_lo && field.vx[n] <= x_hi
            && field.vy[n] >= y_lo && field.vy[n] <= y_hi
    };

    let tri_sel: Vec<usize> = (0..field.t0.len())
        .filter(|&k| inside(field.t0[k]) && inside(field.t1[k]) && inside(field.t2[k]))
        .collect();

    let mut used: HashSet<usize> = HashSet::new();
    for &k in &tri_sel {
        used.insert(field.t0[k]);
        used.insert(field.t1[k]);
        used.insert(field.t2[k]);
    }
    let mut used_sorted: Vec<usize> = used.into_iter().collect();
    used_sorted.sort_unstable();

    let mut remap = vec![0usize; field.vx.len()];
    for (new_idx, &old_idx) in used_sorted.iter().enumerate() {
        remap[old_idx] = new_idx;
    }

    FilteredMesh {
        vx:       used_sorted.iter().map(|&i| field.vx[i]).collect(),
        vy:       used_sorted.iter().map(|&i| field.vy[i]).collect(),
        t0:       tri_sel.iter().map(|&k| remap[field.t0[k]]).collect(),
        t1:       tri_sel.iter().map(|&k| remap[field.t1[k]]).collect(),
        t2:       tri_sel.iter().map(|&k| remap[field.t2[k]]).collect(),
        vel_x:    tri_sel.iter().map(|&k| field.vel_x[k]).collect(),
        vel_y:    tri_sel.iter().map(|&k| field.vel_y[k]).collect(),
        pressure: tri_sel.iter().map(|&k| field.pressure[k]).collect(),
    }
}
