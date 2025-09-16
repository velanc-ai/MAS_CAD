import argparse
import itertools
import numpy as np
import trimesh
import sys, os
from time import time
import pandas as pd

# ---------- helpers ----------

def log(msg):
    print(msg, flush=True)

def bail(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)

def to_principal_axes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    if not m.is_watertight:
        try:
            m.remove_degenerate_faces()
            m.remove_unreferenced_vertices()
            if not m.is_watertight:
                m = m.fill_holes()
        except Exception:
            pass
    T = m.principal_inertia_transform
    m.apply_transform(T)
    return m

def candidate_rotations():
    mats = []
    I = np.eye(3)
    for perm in itertools.permutations(range(3)):
        P = I[:, perm]
        for s in itertools.product([-1, 1], repeat=3):
            S = np.diag(s)
            R = P @ S
            if np.linalg.det(R) > 0.999:
                mats.append(R)
    uniq = []
    for R in mats:
        if not any(np.allclose(R, U, atol=1e-8) for U in uniq):
            uniq.append(R)
    return uniq

def centroid_align(mesh, target=np.zeros(3)):
    m = mesh.copy()
    m.apply_translation(target - m.centroid)
    return m

def voxel_iou_shared_grid(mesh_a, mesh_b, grid=128):
    stack = trimesh.util.concatenate([mesh_a, mesh_b])
    max_dim = float(np.max(stack.extents))
    if max_dim <= 0:
        return 0.0
    pitch = max_dim / max(grid, 2)
    origin = stack.bounds[0]

    va = mesh_a.voxelized(pitch=pitch).fill()
    vb = mesh_b.voxelized(pitch=pitch).fill()

    pa = va.points
    pb = vb.points

    if len(pa) == 0 and len(pb) == 0:
        return 1.0
    if len(pa) == 0 or len(pb) == 0:
        return 0.0

    ia = np.floor((pa - origin) / pitch + 0.5).astype(np.int32)
    ib = np.floor((pb - origin) / pitch + 0.5).astype(np.int32)

    occ_a = {tuple(x) for x in ia}
    occ_b = {tuple(x) for x in ib}

    inter = len(occ_a & occ_b)
    union = len(occ_a | occ_b)
    return (inter / union) if union > 0 else 0.0

def best_align(gen_mesh, gt_mesh, probe_grid=48):
    log("→ Aligning to principal axes …")
    gt_pa = centroid_align(to_principal_axes(gt_mesh))
    gen_pa = centroid_align(to_principal_axes(gen_mesh))

    log("→ Searching best orientation (24 candidates) …")
    best_iou, best_gen = -1.0, None
    for idx, R in enumerate(candidate_rotations(), start=1):
        cand = gen_pa.copy()
        T = np.eye(4)
        T[:3, :3] = R
        cand.apply_transform(T)
        cand = centroid_align(cand)

        score = voxel_iou_shared_grid(cand, gt_pa, grid=probe_grid)
        if score > best_iou:
            best_iou, best_gen = score, cand
        if idx % 6 == 0:
            log(f"   … tried {idx}/24, current best (probe) IoU={best_iou:.4f}")

    log(f"✓ Best probe IoU={best_iou:.4f}")
    return best_gen, gt_pa

def iou_3d(path_a, path_b, grid=128):
    t0 = time()

    if not os.path.exists(path_a):
        bail(f"File not found for --a: {path_a}")
    if not os.path.exists(path_b):
        bail(f"File not found for --b: {path_b}")

    try:
        size_a = os.path.getsize(path_a)
        size_b = os.path.getsize(path_b)
        log(f"Loading meshes:\n  A: {path_a} ({size_a/1e6:.2f} MB)\n  B: {path_b} ({size_b/1e6:.2f} MB)")
    except Exception:
        log(f"Loading meshes:\n  A: {path_a}\n  B: {path_b}")

    try:
        a = trimesh.load(path_a, force='mesh')
    except Exception as e:
        bail(f"Failed to load A: {e}")
    try:
        b = trimesh.load(path_b, force='mesh')
    except Exception as e:
        bail(f"Failed to load B: {e}")

    if len(a.faces) > 50000:
        log("→ Simplifying mesh A (high face count)…")
        a = a.simplify_quadratic_decimation(target_faces=50000)
    if len(b.faces) > 50000:
        log("→ Simplifying mesh B (high face count)…")
        b = b.simplify_quadratic_decimation(target_faces=50000)

    a_aligned, b_aligned = best_align(a, b, probe_grid=min(64, max(24, grid//2)))
    log(f"→ Computing final voxel IoU on grid={grid} …")
    iou = voxel_iou_shared_grid(a_aligned, b_aligned, grid=grid)
    elapsed = time() - t0
    log(f"✓ Done in {elapsed:.2f}s")
    return iou, elapsed

def log_to_excel(filename_a, filename_b, grid, iou_score, iou_percent, execution_time, output_file="iou_results.xlsx"):
    """Append IoU result to an Excel file with rounded precision and consistent column names."""
    # Round to exactly 2 decimal places for percentage and execution time
    iou_percent_rounded = round(iou_percent, 2)
    execution_time_rounded = round(execution_time, 2)

    data = {
        "MAS_CAD": [os.path.basename(filename_a)],      # ← Changed from "File A"
        "Benchmark_CAD": [os.path.basename(filename_b)], # ← Changed from "File B"
        "Grid Size": [grid],
        "IoU Score": [iou_score],
        "IoU Percentage (%)": [iou_percent_rounded],
        "Execution Time (s)": [execution_time_rounded],
        "Timestamp": [pd.Timestamp.now()]
    }

    df = pd.DataFrame(data)

    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False, header=False, startrow=writer.sheets['Results'].max_row)
    else:
        df.to_excel(output_file, sheet_name='Results', index=False)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Fast 3D IoU (principal axes + centroid + voxelization). Optimized for small STLs."
    )
    ap.add_argument("--a", type=str, default=None, help="Path to model A (prediction) - use with --b")
    ap.add_argument("--b", type=str, default=None, help="Path to model B (ground truth) - use with --a")
    ap.add_argument("--grid", type=int, default=128,
                    help="Voxel grid along longest bbox side. Default=128.")
    ap.add_argument("--output", type=str, default="iou_results.xlsx",
                    help="Output Excel file path. Default=iou_results.xlsx")
    ap.add_argument("--csv", type=str, default=None,
                    help="CSV file with columns 'MAS_CAD' and 'Benchmark_CAD'. Overrides --a/--b.")

    args = ap.parse_args()

    if args.csv:
        # BATCH MODE: Process all rows in CSV
        if not os.path.exists(args.csv):
            bail(f"CSV file not found: {args.csv}")

        log(f"🔄 Running in batch mode using CSV: {args.csv}")
        df = pd.read_csv(args.csv)

        required_cols = {'MAS_CAD', 'Benchmark_CAD'}
        if not required_cols.issubset(df.columns):
            bail(f"CSV must contain columns: {required_cols}. Found: {list(df.columns)}")

        total = len(df)
        log(f"Found {total} pairs to process...")

        for idx, row in df.iterrows():
            a_path = row['MAS_CAD']
            b_path = row['Benchmark_CAD']

            log(f"\n--- Processing {idx+1}/{total}:")
            log(f"  MAS_CAD: {a_path}")
            log(f"  Benchmark_CAD: {b_path}")

            try:
                iou, elapsed = iou_3d(a_path, b_path, grid=args.grid)
                print(f"IoU: {iou:.6f}  ({iou*100:.2f}%) in {elapsed:.2f}s")

                log_to_excel(
                    filename_a=a_path,
                    filename_b=b_path,
                    grid=args.grid,
                    iou_score=iou,
                    iou_percent=iou * 100,
                    execution_time=elapsed,
                    output_file=args.output
                )

                log(f"✓ Result logged to: {args.output}")

            except Exception as e:
                log(f"❌ FAILED: {e}")
                continue  # Skip this pair and keep going

        log(f"\n✅ Batch processing complete. Results saved to '{args.output}'")

    else:
        # SINGLE MODE: Use --a and --b
        if not args.a or not args.b:
            bail("Either --csv must be provided OR both --a and --b must be specified.")

        log(f"Args: A='{args.a}'  B='{args.b}'  grid={args.grid}  output='{args.output}'")

        try:
            iou, elapsed = iou_3d(args.a, args.b, grid=args.grid)
            print(f"IoU: {iou:.6f}  ({iou*100:.2f}%) in {elapsed:.2f}s")

            log_to_excel(
                filename_a=args.a,
                filename_b=args.b,
                grid=args.grid,
                iou_score=iou,
                iou_percent=iou * 100,
                execution_time=elapsed,
                output_file=args.output
            )

            log(f"✓ Result logged to: {args.output}")

        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr, flush=True)
            sys.exit(130)

if __name__ == "__main__":
    main()