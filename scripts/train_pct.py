#!/usr/bin/env python
"""
Thin launcher for qq456cvb/Point-Transformers classification training.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback # noqa
from pathlib import Path

# --- paths ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'repos' / 'Point-Transformers'
DATA_DEFAULT = ROOT / 'datasets' / 'modelnet40_normal_resampled'


def run_and_report(cmd, cwd, env=None):
    """Run a command, streaming output.
    On failure, print captured stderr/stdout.
    """
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
    except subprocess.CalledProcessError as e:
        print('\n[ERROR] Subprocess failed:')
        print('  CWD:', cwd)
        print('  CMD:', ' '.join(map(str, e.cmd)))
        print('  RETURN CODE:', e.returncode)
        # Suggest direct run for deeper trace
        print('\nTip: run this directly in a cell for full traceback:')
        print(f"%cd {cwd}\n!python -u {' '.join(map(str, cmd[1:]))}")
        raise


def ensure_symlink(target: Path, link: Path) -> None:
    """Create/refresh `link` -> `target` symlink if needed."""
    link_parent = link.parent
    link_parent.mkdir(parents=True, exist_ok=True)
    try:
        if link.exists() or link.is_symlink():
            # If it points somewhere else, replace it
            if link.resolve() != target.resolve():
                link.unlink()
                link.symlink_to(target, target_is_directory=True)
        else:
            link.symlink_to(target, target_is_directory=True)
    except OSError as e:
        print(f'[warn] Could not create symlink {link} -> {target}: {e}')


def git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=path
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
        'extra',
        nargs=argparse.REMAINDER,
        help='Pass-through args for train_cls.py (Hydra overrides etc). '
        'Prefix with -- to separate, e.g. -- trainer.fast_dev_run=1'
    )
    ap.add_argument(
        '--no-symlink',
        action='store_true',
        help='Skip creating dataset symlink inside the repo'
    )
    args = ap.parse_args()

    # 1) decide output directory
    out = args.out or (ROOT / 'runs' / 'pct' / time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)

    # 2) provenance
    meta = {
        'model': 'Point-Transformers (PCT variant)',
        'repo_path': str(REPO),
        'repo_commit': git_rev(REPO),
        'bench_commit': git_rev(ROOT),
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'argv': sys.argv,
    }
    # Try to capture torch/cuda if available in this env
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

    # 3) dataset symlink
    #    (so hardcoded repo paths like ../../datasets/... resolve)
    #    The repo often expects a folder named 'modelnet40_normal_resampled'
    #    in its root.
    if not args.no_symlink:
        ensure_symlink(args.data, REPO / 'modelnet40_normal_resampled')

    # 4) build command: forward all overrides transparently
    #    You can put hydra overrides after '--', e.g. optimizer.lr=0.001
    #    If you just want help: scripts/train_pct.py -- --help
    cmd = [sys.executable, 'train_cls.py', *args.extra]
    print('>>> CWD:', REPO)
    print('>>>', ' '.join(map(str, cmd)))

    # 5) run from the repo root
    run_and_report(cmd, cwd=REPO)


if __name__ == '__main__':
    main()
