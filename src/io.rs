use std::fs::File;
use std::io::Write;

use crate::filtered_mesh::{FilteredMesh, StokesField};

// ---------------------------------------------------------------------------
// VTK reader
// ---------------------------------------------------------------------------

/// Read a legacy ASCII VTK file that contains a triangle mesh with per-cell
/// `velocity` (vector) and `pressure` (scalar) fields.
pub fn read_vtk_2d(path: &str) -> StokesField {
    let text = std::fs::read_to_string(path).expect("failed to read VTK file");
    let lines: Vec<&str> = text.lines().map(str::trim).collect();

    let pi = lines.iter().position(|l| l.starts_with("POINTS")).unwrap();
    let n_points: usize = lines[pi].split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut vx = Vec::with_capacity(n_points);
    let mut vy = Vec::with_capacity(n_points);
    for k in 0..n_points {
        let mut it = lines[pi + 1 + k].split_whitespace();
        vx.push(it.next().unwrap().parse::<f64>().unwrap());
        vy.push(it.next().unwrap().parse::<f64>().unwrap());
    }

    let ci = lines.iter().position(|l| l.starts_with("CELLS")).unwrap();
    let n_cells: usize = lines[ci].split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut t0 = Vec::with_capacity(n_cells);
    let mut t1 = Vec::with_capacity(n_cells);
    let mut t2 = Vec::with_capacity(n_cells);
    for k in 0..n_cells {
        let mut it = lines[ci + 1 + k].split_whitespace();
        assert_eq!(it.next().unwrap(), "3", "expected triangles");
        t0.push(it.next().unwrap().parse::<usize>().unwrap());
        t1.push(it.next().unwrap().parse::<usize>().unwrap());
        t2.push(it.next().unwrap().parse::<usize>().unwrap());
    }

    let vi = lines.iter().position(|l| l.starts_with("velocity")).unwrap();
    let mut vel_x = Vec::with_capacity(n_cells);
    let mut vel_y = Vec::with_capacity(n_cells);
    for k in 0..n_cells {
        let mut it = lines[vi + 1 + k].split_whitespace();
        vel_x.push(it.next().unwrap().parse::<f64>().unwrap());
        vel_y.push(it.next().unwrap().parse::<f64>().unwrap());
    }

    let pri = lines.iter().position(|l| l.starts_with("pressure")).unwrap();
    let pressure: Vec<f64> = (0..n_cells)
        .map(|k| lines[pri + 1 + k].parse().unwrap())
        .collect();

    StokesField { vx, vy, t0, t1, t2, vel_x, vel_y, pressure }
}

// ---------------------------------------------------------------------------
// VTK writers
// ---------------------------------------------------------------------------

/// Write a particle snapshot (POLYDATA) with per-point velocity and pressure.
pub fn write_vtk_particles(
    path: &str,
    step: usize,
    px: &[f64],
    py: &[f64],
    vel_x: &[f64],
    vel_y: &[f64],
    pressure: &[f64],
) {
    let n = px.len();
    let mut f = File::create(path).unwrap();

    writeln!(f, "# vtk DataFile Version 2.0").unwrap();
    writeln!(f, "Particles step {}", step).unwrap();
    writeln!(f, "ASCII").unwrap();
    writeln!(f, "DATASET POLYDATA").unwrap();
    writeln!(f, "POINTS {} double", n).unwrap();
    for i in 0..n {
        writeln!(f, "{:.9} {:.9} 0.0", px[i], py[i]).unwrap();
    }
    writeln!(f, "VERTICES {} {}", n, 2 * n).unwrap();
    for i in 0..n {
        writeln!(f, "1 {}", i).unwrap();
    }
    writeln!(f, "POINT_DATA {}", n).unwrap();
    writeln!(f, "SCALARS pressure double 1").unwrap();
    writeln!(f, "LOOKUP_TABLE default").unwrap();
    for i in 0..n {
        writeln!(f, "{:.9e}", pressure[i]).unwrap();
    }
    writeln!(f, "VECTORS velocity double").unwrap();
    for i in 0..n {
        writeln!(f, "{:.9e} {:.9e} 0.0", vel_x[i], vel_y[i]).unwrap();
    }
}

/// Write a filtered sub-mesh (UNSTRUCTURED_GRID) with per-cell velocity and
/// pressure.
pub fn write_vtk_submesh(path: &str, fmesh: &FilteredMesh) {
    let n_pts   = fmesh.vx.len();
    let n_cells = fmesh.t0.len();

    let mut f = File::create(path).unwrap();
    writeln!(f, "# vtk DataFile Version 2.0").unwrap();
    writeln!(f, "Acoustic mesh under study").unwrap();
    writeln!(f, "ASCII").unwrap();
    writeln!(f, "DATASET UNSTRUCTURED_GRID").unwrap();

    writeln!(f, "POINTS {} double", n_pts).unwrap();
    for i in 0..n_pts {
        writeln!(f, "{:.9} {:.9} 0.0", fmesh.vx[i], fmesh.vy[i]).unwrap();
    }

    writeln!(f, "CELLS {} {}", n_cells, n_cells * 4).unwrap();
    for i in 0..n_cells {
        writeln!(f, "3 {} {} {}", fmesh.t0[i], fmesh.t1[i], fmesh.t2[i]).unwrap();
    }

    writeln!(f, "CELL_TYPES {}", n_cells).unwrap();
    for _ in 0..n_cells {
        writeln!(f, "5").unwrap();
    }

    writeln!(f, "CELL_DATA {}", n_cells).unwrap();
    writeln!(f, "SCALARS pressure double 1").unwrap();
    writeln!(f, "LOOKUP_TABLE default").unwrap();
    for i in 0..n_cells {
        writeln!(f, "{:.9e}", fmesh.pressure[i]).unwrap();
    }
    writeln!(f, "VECTORS velocity double").unwrap();
    for i in 0..n_cells {
        writeln!(f, "{:.9e} {:.9e} 0.0", fmesh.vel_x[i], fmesh.vel_y[i]).unwrap();
    }

    println!("wrote {} ({} points, {} triangles)", path, n_pts, n_cells);
}

// ---------------------------------------------------------------------------
// ParaView helpers
// ---------------------------------------------------------------------------

/// Write a `visualize.py` script into `out_dir` that loads the static field
/// mesh and animates the particle snapshots in ParaView.
pub fn write_visualize_py(out_dir: &str, field_vtk_rel: &str, n_particles: usize) {
    let path = format!("{}/visualize.py", out_dir);
    let mut f = File::create(&path).unwrap();

    write!(
        f,
        r#"# Auto-generated by jaali_clow
# Run from the "{out_dir}" directory:
#   pvpython visualize.py
#
# Simulation: Acoustic 2D — {n_particles} particles

from paraview.simple import *
import glob, re

paraview.simple._DisableFirstRenderCameraReset()

_SIM = "Acoustic 2D — {n_particles} particles"

# ── Background acoustic field ────────────────────────────────
field_reader = LegacyVTKReader(
    registrationName='acoustic_field',
    FileNames=['{field_vtk_rel}'])
field_reader.UpdatePipeline()

view = GetActiveViewOrCreate('RenderView')

field_disp = Show(field_reader, view, 'UnstructuredGridRepresentation')
field_disp.SetRepresentationType('Surface')
ColorBy(field_disp, ('CELLS', 'velocity'))
lut_field = GetColorTransferFunction('velocity')
lut_field.VectorMode = 'Magnitude'
field_disp.RescaleTransferFunctionToDataRange(False, True)
lut_field.ApplyPreset('Cool to Warm', True)
field_disp.SetScalarBarVisibility(view, True)

bar_field = GetScalarBar(lut_field, view)
bar_field.Orientation        = 'Horizontal'
bar_field.WindowLocation     = 'Any Location'
bar_field.Position           = [0.25, 0.18]
bar_field.ScalarBarLength    = 0.50
bar_field.ScalarBarThickness = 16
bar_field.Title              = 'velocity magnitude'

# ── Velocity vectors (arrows) ────────────────────────────────
c2p = CellDatatoPointData(registrationName='c2p', Input=field_reader)
c2p.ProcessAllArrays = 1
c2p.UpdatePipeline()

glyph = Glyph(registrationName='vel_arrows', Input=c2p, GlyphType='Arrow')
glyph.OrientationArray  = ['POINTS', 'velocity']
glyph.ScaleArray        = ['POINTS', 'velocity']
glyph.ScaleFactor       = 1.0
glyph.GlyphMode         = 'Every Nth Point'
glyph.Stride            = 20
glyph.UpdatePipeline()

glyph_disp = Show(glyph, view, 'GeometryRepresentation')
ColorBy(glyph_disp, ('POINTS', 'velocity'))
lut_g = GetColorTransferFunction('velocity')
lut_g.VectorMode = 'Magnitude'
glyph_disp.RescaleTransferFunctionToDataRange(False, True)
glyph_disp.SetScalarBarVisibility(view, False)

# ── Particle animation ───────────────────────────────────────
p_files = sorted(
    glob.glob('particles_*.vtk'),
    key=lambda f: int(re.search(r'\d+', f).group()))
if not p_files:
    raise RuntimeError(
        'No particles_*.vtk found. Run the simulation first, '
        'then launch pvpython from the vtk_out directory.')
print(f'[{{_SIM}}] {{len(p_files)}} timestep(s): {{p_files[0]}} ... {{p_files[-1]}}')

p_reader = LegacyVTKReader(registrationName='particles', FileNames=p_files)
p_reader.UpdatePipeline()

GetAnimationScene().UpdateAnimationUsingDataTimeSteps()

p_disp = Show(p_reader, view, 'UnstructuredGridRepresentation')
p_disp.SetRepresentationType('Point Gaussian')
p_disp.GaussianRadius   = 2.0
p_disp.ScaleByArray     = 0
p_disp.UseScaleFunction = 0
ColorBy(p_disp, ('POINTS', 'pressure'))
lut_p = GetColorTransferFunction('pressure')
p_disp.RescaleTransferFunctionToDataRangeOverTime()
lut_p.ApplyPreset('Plasma (matplotlib)', True)
p_disp.SetScalarBarVisibility(view, True)

bar_p = GetScalarBar(lut_p, view)
bar_p.Orientation        = 'Horizontal'
bar_p.WindowLocation     = 'Any Location'
bar_p.Position           = [0.25, 0.02]
bar_p.ScalarBarLength    = 0.50
bar_p.ScalarBarThickness = 16
bar_p.Title              = 'pressure (particles)'

# ── Camera — look down Z at the 2-D plane ────────────────────
_b  = field_reader.GetDataInformation().GetBounds()
_cx = 0.5 * (_b[0] + _b[1])
_cy = 0.5 * (_b[2] + _b[3])
_sp = max(_b[1] - _b[0], _b[3] - _b[2]) * 0.75

view.CameraPosition   = [_cx, _cy, _sp * 3.0]
view.CameraFocalPoint = [_cx, _cy, 0.0]
view.CameraViewUp     = [0.0, 1.0, 0.0]
view.ResetCamera()
Render()
print(f'[{{_SIM}}] visualization ready.')
"#,
        out_dir = out_dir,
        field_vtk_rel = field_vtk_rel,
        n_particles = n_particles,
    )
    .unwrap();

    println!("wrote {}", path);
}

/// Write a `reload_macro.py` ParaView macro into `out_dir` that reloads all
/// VTK readers from disk without restarting ParaView.
pub fn write_reload_macro_py(out_dir: &str) {
    let path = format!("{}/reload_macro.py", out_dir);
    let mut f = File::create(&path).unwrap();

    write!(
        f,
        r#""""
jaali_clow — ParaView reload macro
====================================

Add this file as a ParaView macro once:
    Tools > Macros > Add New Macro > (select this file)

Then trigger it whenever you want to pick up newly written VTK files.

What it does
------------
1. Finds every LegacyVTKReader currently loaded in the pipeline.
2. Derives the glob pattern from the first file in each reader's list.
3. Re-scans the same directory, sorts numerically, skips empty/corrupt files,
   and updates each reader.
4. Refreshes the animation timeline and re-renders.
""""

from paraview.simple import *
import glob, os, re

_VTK_READER_NAMES = {{
    'LegacyVTKReader',
    'PVLegacyVTKFileReader',
    'legacyVTKReader',
}}

_SIM = "jaali_clow"


def _is_vtk_reader(proxy):
    if proxy.GetXMLName() in _VTK_READER_NAMES:
        return True
    try:
        files = list(proxy.FileNames)
        return bool(files) and files[0].endswith('.vtk')
    except Exception:
        return False


def _is_valid_vtk(filepath):
    try:
        return os.path.getsize(filepath) > 0
    except OSError:
        return False


def _sorted_valid_vtk(pattern):
    files = glob.glob(pattern)
    files.sort(key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))
    valid = []
    for fp in files:
        if _is_valid_vtk(fp):
            valid.append(fp)
        else:
            print(f'[{{_SIM}}] WARNING: skipping empty/corrupt file {{fp}}')
    return valid


def _make_pattern(filepath):
    abspath   = os.path.abspath(filepath)
    directory = os.path.dirname(abspath)
    basename  = os.path.basename(abspath)
    m = re.match(r'^(.*?)_\d+\.vtk$', basename)
    if m is None:
        return None
    return os.path.join(directory, f'{{m.group(1)}}_*.vtk')


try:
    sources = GetSources()
    if not sources:
        print(f'[{{_SIM}}] Pipeline is empty — run visualize.py first.')
    else:
        print(f'[{{_SIM}}] Pipeline has {{len(sources)}} source(s).')

    reloaded = 0

    for proxy in sources.values():
        if not _is_vtk_reader(proxy):
            continue
        try:
            current_files = list(proxy.FileNames)
        except Exception:
            continue
        if not current_files:
            continue
        pattern = _make_pattern(current_files[0])
        if pattern is None:
            continue
        new_files = _sorted_valid_vtk(pattern)
        if not new_files:
            continue
        proxy.FileNames = new_files
        proxy.UpdatePipeline()
        label = proxy.GetXMLLabel() or pattern
        print(f'[{{_SIM}}] Reloaded {{label}}: {{len(new_files)}} file(s).')
        reloaded += 1

    if reloaded == 0:
        print(f'[{{_SIM}}] No VTK readers updated.')
    else:
        GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
        Render()
        print(f'[{{_SIM}}] Done — {{reloaded}} reader(s) refreshed.')

except Exception as e:
    print(f'[{{_SIM}}] Reload failed: {{e}}')
    raise
"#
    )
    .unwrap();

    println!("wrote {}", path);
}
