#!/usr/bin/env python3
"""
Run full decision demo pipeline end-to-end.

  python run_pipeline.py              # use real head200 if present, else mock
  python run_pipeline.py --use-mock   # force mock data (for testing)
  python run_pipeline.py --multperiod   # also run multi-period knapsack
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / '.venv' / 'bin' / 'python'


def _python() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [_python(), str(ROOT / script)] + (extra or [])
    print('\n>>', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description='Hit Song Intelligence — decision pipeline')
    ap.add_argument('--csv', type=str, default=None, help='head200 CSV path')
    ap.add_argument('--base-row', type=int, default=0)
    ap.add_argument('--use-mock', action='store_true', help='Create/use mock head200')
    ap.add_argument('--multperiod', action='store_true', help='Run multi-period knapsack')
    args = ap.parse_args()

    extra = []
    if args.csv:
        extra += ['--csv', args.csv]
    if args.base_row:
        extra += ['--base-row', str(args.base_row)]
    if args.use_mock:
        _run('create_mock_head200.py')
        extra.append('--use-mock')

    from paths import MOCK_CSV, resolve_head200_csv
    try:
        resolve_head200_csv(args.csv)
        has_csv = True
    except FileNotFoundError:
        has_csv = False
    if args.use_mock or (not has_csv and not args.csv):
        if not MOCK_CSV.exists():
            _run('create_mock_head200.py')
        if '--use-mock' not in extra:
            extra.append('--use-mock')

    _run('make_synthetic_new_songs.py', extra)
    _run('score_synthetic_songs.py', extra)
    _run('run_knapsack_demo.py')
    if args.multperiod:
        _run('run_knapsack_multperiod.py')

    print('\n=== Pipeline done ===')
    print('Outputs in decision/outputs/')


if __name__ == '__main__':
    main()
