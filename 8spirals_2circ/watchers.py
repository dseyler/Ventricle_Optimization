#!/usr/bin/env python3
"""
Watchers for running simulations:
- NaN watcher: tails histor.dat and exits when a NaN appears
- Volume watcher: computes volume over time and exits when volume exceeds a threshold factor of the undeformed volume

Usage examples:
  python3 watchers.py nan --results-glob "results_ventricle_8_20_c_p15*" --marker .nan_detected
  python3 watchers.py volume --ref-surface meshes/ventricle_8_20_c-mesh-complete/mesh-surfaces/epi.vtp \
                           --results-glob "results_ventricle_8_20_c_p15*" --threshold-factor 3.0 --marker .volume_exceeded
"""

import argparse
import glob
import os
import sys
import time


def wait_for_latest_histor(results_glob: str, poll_interval: float = 2.0) -> str:
    while True:
        folders = glob.glob(results_glob)
        if folders:
            folders.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            latest = folders[0]
            histor = os.path.join(latest, 'histor.dat')
            if os.path.isfile(histor):
                return histor
        time.sleep(poll_interval)


def watch_nan(results_glob: str, marker: str, poll_interval: float = 1.0) -> None:
    histor_path = wait_for_latest_histor(results_glob, poll_interval)
    print(f"[NaNWatcher] Watching {histor_path}")
    try:
        with open(histor_path, 'r', errors='ignore') as f:
            f.seek(0, os.SEEK_END)  # start at end (tail -n 0)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(poll_interval)
                    continue
                if 'nan' in line.lower():
                    print(f"[NaNWatcher] NaN detected: {line.strip()}")
                    open(marker, 'w').close()
                    return
    except Exception as e:
        print(f"[NaNWatcher] Error reading histor.dat: {e}")


def compute_ref_volume(ref_surface_path: str):
    try:
        import pyvista as pv
    except Exception as e:
        print(f"[VolumeWatcher] PyVista not available: {e}")
        return None, None
    try:
        ref = pv.read(ref_surface_path)
        ref_lumen = ref.fill_holes(100)
        ref_lumen.compute_normals(inplace=True)
        return ref_lumen, float(ref_lumen.volume)
    except Exception as e:
        print(f"[VolumeWatcher] Failed to read/prepare reference surface: {e}")
        return None, None


def latest_results_folder(pattern: str):
    folders = glob.glob(pattern)
    if not folders:
        return None
    folders.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return folders[0]


def watch_volume(ref_surface_path: str, results_glob: str, threshold_factor: float, marker: str, poll_interval: float = 5.0, log_file: str | None = None) -> None:
    try:
        import pyvista as pv
    except Exception as e:
        print(f"[VolumeWatcher] PyVista not available: {e}")
        return

    ref_lumen, V0 = compute_ref_volume(ref_surface_path)
    if ref_lumen is None or V0 is None:
        print("[VolumeWatcher] Reference volume unavailable; exiting watcher.")
        return
    threshold = threshold_factor * V0
    init_msg = f"[VolumeWatcher] V0={V0:.6f}, threshold={threshold:.6f}"
    print(init_msg)
    if log_file:
        try:
            with open(log_file, 'a') as lf:
                lf.write(init_msg + "\n")
        except Exception:
            pass

    seen = set()
    while True:
        rf = latest_results_folder(results_glob)
        if rf:
            vtus = glob.glob(os.path.join(rf, 'result_*.vtu'))
            vtus.sort()
            for vtu in vtus:
                if vtu in seen:
                    continue
                seen.add(vtu)
                try:
                    result = pv.read(vtu)
                    resampled = ref_lumen.sample(result)
                    warped = resampled.warp_by_vector('Displacement')
                    vol = float(warped.volume)
                    line = f"[VolumeWatcher] {os.path.basename(vtu)} volume={vol:.6f}"
                    print(line)
                    if log_file:
                        try:
                            with open(log_file, 'a') as lf:
                                lf.write(line + "\n")
                        except Exception:
                            pass
                    if vol > threshold:
                        exceed_msg = f"[VolumeWatcher] Threshold exceeded: {vol:.6f} > {threshold:.6f}"
                        print(exceed_msg)
                        if log_file:
                            try:
                                with open(log_file, 'a') as lf:
                                    lf.write(exceed_msg + "\n")
                            except Exception:
                                pass
                        open(marker, 'w').close()
                        return
                except Exception as e:
                    # Ignore transient read errors while result file is being written
                    pass
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description='Simulation watchers (NaN and Volume).')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    p_nan = subparsers.add_parser('nan', help='Watch histor.dat for NaNs')
    p_nan.add_argument('--results-glob', required=True, help='Glob pattern for results folders (e.g., results_ventricle_8_20_c_p15*)')
    p_nan.add_argument('--marker', required=True, help='Path to marker file to create when condition met')
    p_nan.add_argument('--poll-interval', type=float, default=1.0)

    p_vol = subparsers.add_parser('volume', help='Watch volumes and stop when exceeding threshold')
    p_vol.add_argument('--ref-surface', required=True, help='Path to reference epi.vtp')
    p_vol.add_argument('--results-glob', required=True, help='Glob pattern for results folders')
    p_vol.add_argument('--threshold-factor', type=float, default=3.0, help='Factor times V0 to trigger stop')
    p_vol.add_argument('--marker', required=True, help='Path to marker file to create when condition met')
    p_vol.add_argument('--poll-interval', type=float, default=5.0)
    p_vol.add_argument('--log-file', help='Path to a log file to append volume lines')

    args = parser.parse_args()

    if args.mode == 'nan':
        watch_nan(args.results_glob, args.marker, poll_interval=args.poll_interval)
    elif args.mode == 'volume':
        watch_volume(args.ref_surface, args.results_glob, args.threshold_factor, args.marker, poll_interval=args.poll_interval, log_file=args.log_file)
    else:
        print('Unknown mode')
        sys.exit(1)


if __name__ == '__main__':
    main()


