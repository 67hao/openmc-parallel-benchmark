"""
run_openmc_task.py - Worker chay tren GitHub Actions Runners cho OpenMC Parallel Benchmark
Paper 10.1 (PNE 2022) - Ho tro balanced sweep plan theo account va runner_id
"""
import argparse
import os
import sys
import time
import json
import shutil
import zipfile
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import openmc
try:
    openmc.config['cross_sections'] = None
except Exception:
    pass
from openmc_model import export_openmc_model, COMPOSITION, DENSITY
import gdrive_helper

def parse_args():
    parser = argparse.ArgumentParser(description="OpenMC Parallel Worker for Paper 10.1")
    parser.add_argument("--runner-id", type=int, required=True, help="ID cua runner (1..20)")
    parser.add_argument("--num-runners", type=int, default=20, help="Tong so runner")
    parser.add_argument("--primaries", type=int, default=100000000, help="So hat moi job (mac dinh 10^8)")
    parser.add_argument("--batches", type=int, default=10, help="So batches")
    parser.add_argument("--gdrive-folder", type=str, default="100_iZR6D3CgCJoHZlxWaGaTzfFR6wPyZ")
    parser.add_argument("--plan-file", type=str, default="openmc_plan_180.json")
    parser.add_argument("--github-user", type=str, default=None)
    parser.add_argument("--timeout-hours", type=float, default=5.4)
    return parser.parse_args()

def run_openmc_simulation(job_dir: Path, openmc_bin: str = "openmc"):
    t0 = time.time()
    env = os.environ.copy()
    if not env.get("OPENMC_CROSS_SECTIONS"):
        env["OPENMC_CROSS_SECTIONS"] = "/tmp/nuclear_data/cross_sections.xml"
    res = subprocess.run([openmc_bin], cwd=str(job_dir), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"    [OPENMC BIN ERROR]: {res.stderr.strip()[-300:]}", flush=True)
    return res.returncode == 0, elapsed

def parse_statepoint(sp_path: Path):
    sp = openmc.StatePoint(str(sp_path))
    t_tot = sp.get_tally(name="total_flux")
    t_peak = sp.get_tally(name="peak_flux")

    tot_val = float(t_tot.mean.flatten()[0])
    tot_err = float(t_tot.std_dev.flatten()[0]) / tot_val * 100.0 if tot_val > 0 else 0.0

    peak_val = float(t_peak.mean.flatten()[0])
    peak_err = float(t_peak.std_dev.flatten()[0]) / peak_val * 100.0 if peak_val > 0 else 0.0

    return peak_val, peak_err, tot_val, tot_err

def main():
    args = parse_args()
    print("=" * 80)
    print(f"🚀 STARTING OPENMC PARALLEL RUNNER #{args.runner_id} / {args.num_runners}")
    print(f"Primaries: {args.primaries:,} ({args.batches} batches)")
    print(f"Target Drive Folder: {args.gdrive_folder}")
    print("=" * 80)

    work_dir = Path("/tmp/openmc_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    drive_service = None
    folder_ids = {}
    try:
        os.environ["GDRIVE_FOLDER_ID"] = args.gdrive_folder
        drive_service = gdrive_helper.get_drive_service()
        folder_ids = gdrive_helper.get_or_create_subfolders(drive_service, parent_id=args.gdrive_folder)
        print(">>> [DRIVE] Ket noi Google Drive OpenMC thanh cong!")
    except Exception as e:
        print(f">>> [CANH BAO] Khong the ket noi Google Drive: {e}. Luu local.")

    csv_filename = f"flux_runner_{args.runner_id:02d}.csv"
    csv_file = work_dir / csv_filename

    completed_jobs = set()
    if drive_service and "02_Flux_Data" in folder_ids:
        print(f">>> [RESUME] Dang kiem tra cac job da xong tren Drive ({csv_filename})...")
        completed_jobs = gdrive_helper.get_completed_jobs_from_drive(drive_service, folder_ids["02_Flux_Data"], target_filename=csv_filename)
        print(f">>> [RESUME] Tim thay {len(completed_jobs)} job OpenMC da hoan thanh tren Drive.")

    if csv_file.exists():
        try:
            df_local = pd.read_csv(csv_file)
            if "job_name" in df_local.columns:
                local_done = set(df_local["job_name"].dropna().astype(str).tolist())
                completed_jobs.update(local_done)
        except Exception:
            pass
    else:
        csv_file.write_text("job_name,energy_mev,sample,thickness_cm,peak_flux,peak_err,total_flux,total_err,elapsed_s\n", encoding="utf-8")

    # Load plan
    with open(args.plan_file, "r", encoding="utf-8") as pf:
        full_plan = json.load(pf)

    user_key = args.github_user or os.environ.get("GITHUB_REPOSITORY_OWNER") or ""
    matched_acc = None
    for acc in full_plan:
        if acc.lower() == user_key.lower():
            matched_acc = acc
            break
    if not matched_acc and full_plan:
        matched_acc = list(full_plan.keys())[0]

    raw_my_jobs = full_plan.get(matched_acc, {}).get(str(args.runner_id), [])
    todo_jobs = [j for j in raw_my_jobs if j["job_name"] not in completed_jobs]

    print(f">>> [PLAN-SWEEP] Account: {matched_acc} | Runner #{args.runner_id}")
    print(f">>> Phan bo: {len(raw_my_jobs)} jobs | Da xong: {len(raw_my_jobs) - len(todo_jobs)} | Con lai: {len(todo_jobs)} jobs")

    if not todo_jobs:
        print(f">>> [HOAN THANH] Toan bo {len(raw_my_jobs)} job cua Runner #{args.runner_id} da xong! Thoat.")
        return

    start_all = time.time()
    batch_idx = 1
    batch_outs = []

    for idx, job in enumerate(todo_jobs, 1):
        elapsed_total = time.time() - start_all
        if elapsed_total > (args.timeout_hours * 3600 - 900):
            print(f">>> [CHECKPOINT] Sap dat nguong timeout an toan ({args.timeout_hours}h). Thoat.")
            break

        jname = job["job_name"]
        jdir = work_dir / "inputs" / jname
        jdir.mkdir(parents=True, exist_ok=True)
        s_key = job["sample"].lower()
        e_kev = float(job["energy_kev"])
        e_mev = float(job["energy_mev"])
        th = float(job["thickness_cm"])

        print(f"\n--- [{idx}/{len(todo_jobs)}] Runner #{args.runner_id} dang chay OpenMC: {jname} ({job['sample']} @ {e_kev} keV, th={th} cm) ---", flush=True)

        # Export XML
        export_openmc_model(str(jdir), s_key, e_kev, th, args.primaries, args.batches)

        # Execute OpenMC
        success, job_time = run_openmc_simulation(jdir, openmc_bin="openmc")

        sp_path = jdir / f"statepoint.{args.batches}.h5"
        if success and sp_path.exists() and sp_path.stat().st_size > 0:
            try:
                peak_val, peak_err, tot_val, tot_err = parse_statepoint(sp_path)
                print(f"    -> OK ({job_time:.1f}s) | Peak: {peak_val:.6e} (err: {peak_err:.2f}%) | Tot: {tot_val:.6e}", flush=True)

                with open(csv_file, "a", encoding="utf-8") as f:
                    f.write(f"{jname},{e_mev:.6f},{job['sample']},{th},{peak_val:.8e},{peak_err:.4f},{tot_val:.8e},{tot_err:.4f},{job_time:.1f}\n")

                batch_outs.append(sp_path)
            except Exception as e:
                print(f"    -> LOI PARSE STATEPOINT: {e}", flush=True)
        else:
            print(f"    -> LOI THUC THI OPENMC HOAC THIEU FILE STATEPOINT! ({job_time:.1f}s)", flush=True)

        if len(batch_outs) >= 3:
            if drive_service and "02_Flux_Data" in folder_ids:
                try:
                    gdrive_helper.upload_file_to_folder(drive_service, str(csv_file), folder_ids["02_Flux_Data"])
                    print(f">>> [SYNC] Da dong bo OpenMC CSV len Drive!")
                    batch_outs = []
                    batch_idx += 1
                except Exception as e:
                    print(f">>> [CANH BAO SYNC] {e}")

    # Final sync
    if drive_service and "02_Flux_Data" in folder_ids and csv_file.exists():
        try:
            gdrive_helper.upload_file_to_folder(drive_service, str(csv_file), folder_ids["02_Flux_Data"])
            print(">>> [FINAL SYNC] Da dong bo toan bo ket qua OpenMC len Google Drive 02_Flux_Data!")
        except Exception as e:
            print(f">>> [CANH BAO FINAL SYNC]: {e}")

    print("\n" + "=" * 55)
    print(f"OPENMC RUNNER #{args.runner_id} HOAN THANH DOT CHAY")
    print(f"Tong thoi gian: {time.time() - start_all:.1f}s")
    print(f"File ket qua flux: {csv_filename}")
    print("=" * 55)

if __name__ == "__main__":
    main()
