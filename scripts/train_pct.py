#!/usr/bin/env python
"""
Thin launcher and logger for qq456cvb/Point-Transformers training.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import shlex
from pathlib import Path

# --- paths ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'repos' / 'Point-Transformers'
DATA_DEFAULT = ROOT / 'datasets' / 'modelnet40_normal_resampled'

# Show full Hydra tracebacks
os.environ['HYDRA_FULL_ERROR'] = '1'


def ensure_symlink(target: Path, link: Path) -> None:
    """Create/refresh `link` -> `target` symlink if needed."""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        if link.exists() or link.is_symlink():
            try:
                if link.resolve() != target.resolve():
                    link.unlink()
                    link.symlink_to(target, target_is_directory=True)
            except FileNotFoundError:
                # If the target doesn't exist yet, just relink
                link.unlink(missing_ok=True)
                link.symlink_to(target, target_is_directory=True)
        else:
            link.symlink_to(target, target_is_directory=True)
    except OSError as e:
        print(f"[warn] Could not create symlink {link} -> {target}: {e}")


def git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=path
        ).decode().strip()
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Output directory (default: runs/pct/<timestamp>)'
    )
    ap.add_argument(
        '--data',
        type=Path,
        default=DATA_DEFAULT,
        help='Path to ModelNet40 '
        '(default: datasets/modelnet40_normal_resampled)'
    )
    ap.add_argument(
        '--no-symlink',
        action='store_true',
        help='Skip creating dataset symlink inside the repo'
    )
    ap.add_argument(
        'extra',
        nargs=argparse.REMAINDER,
        help='Pass-through args for train_cls.py (Hydra overrides etc). '
        'You may prefix with --'
    )
    args = ap.parse_args()

    # 1) Decide output directory
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or (ROOT / 'runs' / 'pct' / ts)
    out.mkdir(parents=True, exist_ok=True)

    # 2) Provenance
    meta = {
        'model': 'Point-Transformers (PCT variant)',
        'repo_path': str(REPO),
        'repo_commit': git_rev(REPO),
        'bench_commit': git_rev(ROOT),
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'argv': sys.argv,
        'python': sys.version
    }
    try:
        import torch  # type: ignore
        meta.update({
            'torch': torch.__version__,
            'cuda_version': getattr(torch.version, 'cuda', None),
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, # noqa
        })
    except Exception:
        pass
    (out / 'provenance.json').write_text(json.dumps(meta, indent=2))

    # 3) Dataset symlink so hardcoded repo paths resolve
    if not args.no_symlink:
        ensure_symlink(args.data, REPO / 'modelnet40_normal_resampled')

    # 4) Build trainer command
    extra = list(args.extra)
    if extra and extra[0] == '--':
        extra = extra[1:]
    cmd_list = [sys.executable, '-u', 'train_cls.py', *extra]

    # 5) Run, log
    env = os.environ.copy()
    log_path = (out / 'train.log').resolve()
    os.chdir(REPO)
    script_bin = shutil.which('script')
    if script_bin:
        cmd_str = ' '.join(shlex.quote(c) for c in cmd_list)
        run_cmd = [script_bin, '-q', '-f', str(log_path), '-c', cmd_str]
        print('>>>', ' '.join(run_cmd))
        rc = subprocess.call(run_cmd, env=env)
        sys.exit(rc)
    else:
        print("[warn] 'script' not found; running without tee (no train.log).")
        print('>>>', ' '.join(cmd_list))
        os.execvpe(cmd_list[0], cmd_list, env)


if __name__ == '__main__':
    main()
