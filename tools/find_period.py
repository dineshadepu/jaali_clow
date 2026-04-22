#!/usr/bin/env python3
"""
find_period.py — detect the acoustic oscillation period from a VTK time-series
and extract the file list for the last stable cycle.

Usage:
    python find_period.py <input_folder> [OPTIONS]

Options:
    --out-dir DIR      Copy selected files into DIR (default: print paths only)
    --probe-tri N      Triangle index to sample pressure from (default: 0)
    --min-prominence F Minimum peak height relative to signal range (default: 0.2)
    --plot             Show a matplotlib plot of the pressure signal and peaks

The script expects files named  field-0.vtk, field-1.vtk, ... (or any prefix
ending in -<integer>.vtk) inside <input_folder>.
"""

import argparse
import glob
import os
import re
import sys


# ---------------------------------------------------------------------------
# VTK reader — extracts pressure at a single triangle index
# ---------------------------------------------------------------------------

def read_pressure_at_tri(path: str, tri_idx: int = 0) -> float:
    """Return the pressure value at *tri_idx* from a legacy ASCII VTK file."""
    with open(path, "r") as fh:
        lines = [l.strip() for l in fh]

    # Locate the pressure data section
    pri = None
    for i, line in enumerate(lines):
        if line.startswith("pressure"):
            pri = i
            break
    if pri is None:
        raise ValueError(f"No 'pressure' field found in {path}")

    # Skip the LOOKUP_TABLE line if present (scalar fields have it)
    data_start = pri + 1
    if lines[data_start].startswith("LOOKUP_TABLE"):
        data_start += 1

    return float(lines[data_start + tri_idx])


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(folder: str):
    """Return (sorted_paths, sorted_indices) for field-N.vtk files."""
    pattern = os.path.join(folder, "*.vtk")
    candidates = glob.glob(pattern)

    numbered = []
    for fp in candidates:
        m = re.search(r"-(\d+)\.vtk$", os.path.basename(fp))
        if m:
            numbered.append((int(m.group(1)), fp))

    if not numbered:
        sys.exit(f"ERROR: no files matching *-<N>.vtk found in {folder!r}")

    numbered.sort(key=lambda x: x[0])
    indices = [x[0] for x in numbered]
    paths   = [x[1] for x in numbered]
    return paths, indices


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def find_peaks(signal: list, min_prominence: float = 0.2) -> list:
    """Return indices of local maxima that exceed a prominence threshold."""
    if len(signal) < 3:
        return []

    sig_min = min(signal)
    sig_max = max(signal)
    sig_range = sig_max - sig_min
    threshold = sig_min + min_prominence * sig_range

    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if signal[i] >= threshold:
                peaks.append(i)
    return peaks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect acoustic oscillation period from VTK time-series.")
    parser.add_argument("folder",            help="Folder containing field-N.vtk files")
    parser.add_argument("--out-dir",         default=None,
                        help="Copy selected files into this directory")
    parser.add_argument("--probe-tri", type=int, default=0,
                        help="Triangle index to sample pressure (default: 0)")
    parser.add_argument("--min-prominence", type=float, default=0.2,
                        help="Peak prominence threshold as fraction of signal range (default: 0.2)")
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib pressure + peak plot")
    args = parser.parse_args()

    # 1. Discover files
    paths, file_indices = discover_files(args.folder)
    print(f"found {len(paths)} VTK files  [{file_indices[0]} … {file_indices[-1]}]")

    # 2. Read pressure signal
    print(f"reading pressure at triangle {args.probe_tri} from each file …", flush=True)
    pressure = []
    for p in paths:
        pressure.append(read_pressure_at_tri(p, args.probe_tri))
    print(f"pressure range: [{min(pressure):.4e}, {max(pressure):.4e}]")

    # 3. Find peaks
    peaks = find_peaks(pressure, min_prominence=args.min_prominence)
    if len(peaks) < 2:
        sys.exit(
            f"ERROR: found only {len(peaks)} peak(s) — need at least 2 to determine a period.\n"
            f"Try lowering --min-prominence (currently {args.min_prominence})."
        )
    print(f"peaks at file-indices: {[file_indices[p] for p in peaks]}")

    # 4. Estimate cycle length (average over all consecutive peak pairs)
    gaps = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    cycle_len = sum(gaps) / len(gaps)
    print(f"cycle length: {cycle_len:.2f} frames  "
          f"(min={min(gaps)}, max={max(gaps)}, n_cycles={len(gaps)})")

    # 5. Extract last full cycle: from second-to-last peak up to (but not including) last peak
    last_cycle_paths = paths[peaks[-2] : peaks[-1]]
    start_idx = file_indices[peaks[-2]]
    end_idx   = file_indices[peaks[-1] - 1]
    print(f"\nlast cycle: {len(last_cycle_paths)} files  "
          f"[field-{start_idx}.vtk … field-{end_idx}.vtk]")
    for fp in last_cycle_paths:
        print(f"  {fp}")

    # 6. Optionally copy files to output directory
    if args.out_dir:
        import shutil
        os.makedirs(args.out_dir, exist_ok=True)
        for fp in last_cycle_paths:
            shutil.copy2(fp, os.path.join(args.out_dir, os.path.basename(fp)))
        print(f"\ncopied {len(last_cycle_paths)} files → {args.out_dir}/")

    # 7. Optionally plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not available — skipping plot")
            return

        _, ax = plt.subplots(figsize=(12, 4))
        ax.plot(file_indices, pressure, lw=1.0, label="pressure[tri 0]")
        peak_x = [file_indices[p] for p in peaks]
        peak_y = [pressure[p]     for p in peaks]
        ax.scatter(peak_x, peak_y, color="red", zorder=5, label="peaks")

        # Shade the last cycle
        ax.axvspan(file_indices[peaks[-2]], file_indices[peaks[-1]],
                   alpha=0.15, color="green", label="last cycle")

        ax.set_xlabel("file index")
        ax.set_ylabel("pressure")
        ax.set_title(f"Acoustic pressure — cycle ≈ {cycle_len:.1f} frames")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
