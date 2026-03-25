extern "C" __device__ __forceinline__
double orient2d(
    double ax, double ay,
    double bx, double by,
    double cx, double cy
) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

extern "C" __device__ __forceinline__
bool point_in_triangle_strict(
    double px, double py,
    double ax, double ay,
    double bx, double by,
    double cx, double cy
) {
    const double EPS = 1e-12;

    double v0 = orient2d(px, py, bx, by, cx, cy);
    double v1 = orient2d(ax, ay, px, py, cx, cy);
    double v2 = orient2d(ax, ay, bx, by, px, py);
    double area = orient2d(ax, ay, bx, by, cx, cy);

    if (area > 0.0) {
        return v0 > EPS && v1 > EPS && v2 > EPS;
    } else {
        return v0 < -EPS && v1 < -EPS && v2 < -EPS;
    }
}


extern "C" __device__ __forceinline__
bool point_in_triangle_inclusive(
    double px, double py,
    double ax, double ay,
    double bx, double by,
    double cx, double cy
) {
    const double EPS = 1e-12;

    double v0 = orient2d(px, py, bx, by, cx, cy);
    double v1 = orient2d(ax, ay, px, py, cx, cy);
    double v2 = orient2d(ax, ay, bx, by, px, py);
    double area = orient2d(ax, ay, bx, by, cx, cy);

    double tol = EPS * (fabs(area) + 1.0);

    if (area > 0.0) {
        return v0 >= -tol && v1 >= -tol && v2 >= -tol;
    } else {
        return v0 <= tol && v1 <= tol && v2 <= tol;
    }
}



extern "C" __global__
void locate_triangles_all(
    const double* qx,
    const double* qy,
    int* indices,
    unsigned short* counts,
    int n_queries,
    int H,

    const double* xmin,
    const double* ymin,
    const double* xmax,
    const double* ymax,
    const int* left,
    const int* right,
    const int* tri,

    const double* vx,
    const double* vy,
    const int* t0,
    const int* t1,
    const int* t2
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_queries) return;

    double px = qx[q];
    double py = qy[q];

    int base = q * H;
    int hit_count = 0;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int n = stack[--sp];

        if (px < xmin[n] || px > xmax[n] ||
            py < ymin[n] || py > ymax[n]) {
            continue;
        }

        int tid = tri[n];
        if (tid >= 0) {
            int i0 = t0[tid];
            int i1 = t1[tid];
            int i2 = t2[tid];

            if (point_in_triangle_inclusive(
                    px, py,
                    vx[i0], vy[i0],
                    vx[i1], vy[i1],
                    vx[i2], vy[i2])) {

                if (hit_count < H) {
                    indices[base + hit_count] = tid;
                    hit_count++;
                }
            }
        } else {
            stack[sp++] = left[n];
            stack[sp++] = right[n];
        }
    }

    counts[q] = (unsigned short)hit_count;
}

extern "C" __global__
void select_min_owner(
    const int*            indices,
    const unsigned short* counts,
    int*                  out,
    int                   n_queries,
    int                   H
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_queries) return;

    int c = (int)counts[q];
    if (c == 0) {
        out[q] = -1;
        return;
    }

    int base  = q * H;
    int owner = indices[base];
    for (int i = 1; i < c; i++) {
        int v = indices[base + i];
        if (v < owner) owner = v;
    }
    out[q] = owner;
}

// For each particle, copy the cell-centred field values into per-particle arrays.
// Particles outside the mesh (tid < 0) get zeros.
extern "C" __global__
void gather_cell_data(
    const int*    triangle_ids,
    const double* cell_vel_x,
    const double* cell_vel_y,
    const double* cell_pressure,
    double*       p_vel_x,
    double*       p_vel_y,
    double*       p_pressure,
    int           n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    int tid = triangle_ids[i];
    if (tid < 0) {
        p_vel_x[i]    = 0.0;
        p_vel_y[i]    = 0.0;
        p_pressure[i] = 0.0;
        return;
    }
    p_vel_x[i]    = cell_vel_x[tid];
    p_vel_y[i]    = cell_vel_y[tid];
    p_pressure[i] = cell_pressure[tid];
}

// Explicit-Euler position update.
extern "C" __global__
void advect_euler_2d(
    double*       px,
    double*       py,
    const double* vel_x,
    const double* vel_y,
    double        dt,
    int           n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    px[i] += dt * vel_x[i];
    py[i] += dt * vel_y[i];
}

// Wrap particle positions into [x_min, x_max) x [y_min, y_max).
extern "C" __global__
void wrap_periodic_2d(
    double* px,
    double* py,
    double  x_min, double x_max,
    double  y_min, double y_max,
    int     n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    double lx = x_max - x_min;
    double ly = y_max - y_min;

    px[i] = x_min + fmod(fmod(px[i] - x_min, lx) + lx, lx);
    py[i] = y_min + fmod(fmod(py[i] - y_min, ly) + ly, ly);
}
