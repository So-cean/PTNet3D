#!/usr/bin/env python3
"""
Aggregate results from all phases into three folders: real_A, real_B, fake_B.
Searches recursively under a results root (default: ./results) and copies matching
NIfTI files into aggregated folders using shutil.copy2.
"""
import argparse
import shutil
from pathlib import Path
import sys

EXTS = {'.nii', '.nii.gz'}


def categorize(filename: str):
    """Map a filename to one of the target folders or return None if unknown.
    Matches suffixes produced by test_monai: real_A, real_B, fake_B , fake_external, real_external
    """
    n = filename.lower()
    # Prefer explicit test_monai suffixes
    if 'real_a' in n:
        return 'real_A'
    if 'real_b' in n:
        return 'real_B'
    if 'fake_b' in n:
        return 'fake_B'
    if 'fake_external' in n:
        return 'fake_external' 
    if 'real_external' in n:
        return 'real_external'
    if 'PTNetSynth_external' in n:
        return 'fake_external'
    return None


def collect(results_root: Path, out_root: Path, dry_run: bool = False, overwrite: bool = False):
    """Collect and copy files into real_A, real_B, fake_B folders.

    results_root: root folder containing experiment results (default ./results)
    out_root: destination folder where real_A/real_B/fake_B will be created
    """
    results_root = results_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()

    if not results_root.exists():
        print(f"Results root does not exist: {results_root}")
        return 1

    targets = {"real_external": out_root / "real_external",
               "fake_external": out_root / "fake_external",
               "real_A": out_root / "real_A",
               "real_B": out_root / "real_B",
               "fake_B": out_root / "fake_B"}
    
    for t in targets.values():
        if not dry_run:
            t.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    # Walk recursively for files that look like result images
    for p in results_root.rglob('*'):
        if p.is_file():
            name = p.name
            lower = name.lower()
            # check extension
            if not any(lower.endswith(ext) for ext in EXTS):
                continue
            group = categorize(name)
            if group is None:
                # skip files that do not match naming convention
                skipped += 1
                continue

            dest_dir = targets[group]
            dest = dest_dir / name

            if dest.exists():
                if overwrite:
                    pass
                else:
                    print(f"Skipping existing: {dest}")
                    skipped += 1
                    continue

            print(f"Copying: {p} -> {dest}")
            if not dry_run:
                # ensure parent exists
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(p), str(dest))
            copied += 1

    print(f"Done. Copied: {copied}, Skipped/unmatched: {skipped}")
    print(f"Aggregated folders created at: {out_root}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description='Aggregate results into real_A, real_B, fake_B')
    p.add_argument('--results_root', default='results', help='Root results directory to scan (default: ./results)')
    p.add_argument('--out_root', default='results_agg', help='Output folder to store aggregated folders (default: ./results_agg)')
    p.add_argument('--dry_run', action='store_true', help='List actions without copying')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing files in destination')
    return p.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    out_root = Path(args.out_root)
    rc = collect(results_root, out_root, dry_run=args.dry_run, overwrite=args.overwrite)
    sys.exit(rc)


if __name__ == '__main__':
    main()
    
'''
python results_agg.py \
    --results_root /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/K2E2_pred/ \
    --out_root /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/K2E2_pred/agg \
    --dry_run
'''

'''
python results_agg.py \
    --results_root /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/BCP2CBCP_pred/ \
    --out_root /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/BCP2CBCP_pred/agg
'''