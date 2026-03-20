#!/usr/bin/env python3
"""
Script to clean up model checkpoints, keeping only:
- The last model (highest iteration)
- Models at multiples of 5000 iterations
- Exported models (in exported/ folders)
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple


def extract_iteration(filename: str) -> int:
    """Extract iteration number from model filename like 'model_15000.pt'"""
    match = re.match(r"model_(\d+)\.pt", filename)
    if match:
        return int(match.group(1))
    return -1


def find_models_to_keep(model_files: List[Tuple[int, Path]]) -> set:
    """Determine which models to keep based on criteria"""
    if not model_files:
        return set()

    keep_models = set()

    # Find the last model (highest iteration)
    last_iteration, last_path = max(model_files, key=lambda x: x[0])
    keep_models.add(last_path)
    print(f"    Keep last model: {last_path.name} (iteration {last_iteration})")

    # Keep models at multiples of 5000
    for iteration, path in model_files:
        if iteration % 5000 == 0:
            keep_models.add(path)
            print(f"    Keep multiple of 5000: {path.name} (iteration {iteration})")

    return keep_models


def clean_models_in_directory(run_dir: Path, dry_run: bool = True) -> Tuple[int, int]:
    """
    Clean models in a single training run directory.
    Returns (num_deleted, num_kept) tuple.
    """
    # Find all model files (excluding exported folder)
    model_files = []
    for file_path in run_dir.glob("model_*.pt"):
        # Skip if it's in the exported folder
        if "exported" in file_path.parts:
            continue

        iteration = extract_iteration(file_path.name)
        if iteration >= 0:
            model_files.append((iteration, file_path))

    if not model_files:
        return 0, 0

    print(f"\n  Processing: {run_dir.relative_to(run_dir.parent.parent)}")
    print(f"  Found {len(model_files)} model files")

    # Determine which models to keep
    keep_models = find_models_to_keep(model_files)

    # Delete models not in keep list
    num_deleted = 0
    num_kept = 0

    for iteration, path in model_files:
        if path in keep_models:
            num_kept += 1
        else:
            print(f"    {'[DRY RUN] Would delete' if dry_run else 'Deleting'}: {path.name} (iteration {iteration})")
            if not dry_run:
                try:
                    path.unlink()
                    num_deleted += 1
                except Exception as e:
                    print(f"      ERROR: Failed to delete {path.name}: {e}")
            else:
                num_deleted += 1

    return num_deleted, num_kept


def clean_all_models(logs_dir: Path, dry_run: bool = True):
    """Clean models in all training run directories"""
    if not logs_dir.exists():
        print(f"ERROR: Logs directory not found: {logs_dir}")
        return

    print(f"Scanning logs directory: {logs_dir}")
    print(f"Mode: {'DRY RUN (no files will be deleted)' if dry_run else 'LIVE (files will be deleted)'}")
    print("=" * 80)

    total_deleted = 0
    total_kept = 0
    runs_processed = 0

    # Iterate through all experiment directories
    for exp_dir in logs_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # Iterate through all run directories within experiment
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check if this directory has model files
            has_models = any(run_dir.glob("model_*.pt"))
            if not has_models:
                continue

            runs_processed += 1
            num_deleted, num_kept = clean_models_in_directory(run_dir, dry_run)
            total_deleted += num_deleted
            total_kept += num_kept

    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Runs processed: {runs_processed}")
    print(f"  Models kept: {total_kept}")
    print(f"  Models {'would be ' if dry_run else ''}deleted: {total_deleted}")

    if dry_run:
        print("\nThis was a DRY RUN. No files were actually deleted.")
        print("Run with --execute flag to actually delete files.")


def main():
    parser = argparse.ArgumentParser(description="Clean up model checkpoints, keeping only last model, multiples of 5000, and exported models")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Path to logs directory (default: logs)")
    parser.add_argument("--execute", action="store_true", help="Actually delete files (default is dry-run mode)")

    args = parser.parse_args()

    # Get absolute path
    script_dir = Path(__file__).parent
    logs_dir = script_dir / args.logs_dir

    if not logs_dir.exists():
        print(f"ERROR: Logs directory not found: {logs_dir}")
        return

    clean_all_models(logs_dir, dry_run=not args.execute)


if __name__ == "__main__":
    main()
