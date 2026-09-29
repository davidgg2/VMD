"""Full 3-component pipeline: DA-W residual, residual boxes, residual+MOG2.

1. DA-W residual — `analyze_video.py` writes the 6-panel comparison
   (DA-W, Depth Anything V2, motion features, residual flow).
2. Residual boxes — thresholded Farneback residual (default 6 native px,
   min area 40) drawn on that comparison.
3. Residual + MOG2 — side-by-side residual | MOG2 | fused tracks
   (`three_algorithms_movie.py`) using the same residual threshold.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYZE_PYTHON = Path(r'C:\Users\DavidGlickman\.venvs\video-behavior-cuda\Scripts\python.exe')
TRACK_PYTHON = Path(r'D:\TrackAnything\.venv\Scripts\python.exe')
CAR_THRESHOLD = 6.0
CAR_MIN_AREA = 40


def run(python: Path, script: str, extra: list[str], cwd: Path) -> None:
    cmd = [str(python), '-u', str(ROOT / 'scripts' / script), *extra]
    print('$ ' + ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f'{script} failed with exit code {proc.returncode}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--output', default=None,
                   help='Output folder (default outputs/full_pipeline/<stem>)')
    p.add_argument('--threshold-native', type=float, default=CAR_THRESHOLD)
    p.add_argument('--min-area', type=int, default=CAR_MIN_AREA)
    p.add_argument('--skip-daw', action='store_true',
                   help='Skip the GPU DA-W comparison if it already exists')
    args = p.parse_args()
    video = Path(args.video)
    out = Path(args.output) if args.output else ROOT / 'outputs' / 'full_pipeline' / video.stem
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    comparison = out / 'comparison.mp4'
    boxed = out / f'{video.stem}_daw_residual_boxes.mp4'
    three = out / f'{video.stem}_three_algorithms.mp4'

    if not args.skip_daw or not comparison.exists():
        run(ANALYZE_PYTHON, 'analyze_video.py',
            ['--video', str(video), '--output', str(out)], ROOT)
    run(TRACK_PYTHON, 'residual_boxes_on_comparison.py', [
        '--comparison', str(comparison),
        '--source', str(video),
        '--output', str(boxed),
        '--threshold-native', str(args.threshold_native),
        '--min-area', str(args.min_area),
    ], ROOT)
    run(TRACK_PYTHON, 'three_algorithms_movie.py', [
        '--videos', str(video),
        '--output', str(out),
        '--threshold-native', str(args.threshold_native),
        '--min-area', str(args.min_area),
    ], ROOT)

    summary = {
        'source': str(video.resolve()),
        'output': str(out.resolve()),
        'components': {
            '1_daw_residual': str(comparison),
            '2_residual_boxes': str(boxed),
            '3_residual_mog2': str(three),
        },
        'residual_threshold_native_px': args.threshold_native,
        'residual_min_area': args.min_area,
        'elapsed_seconds': time.time() - start,
    }
    (out / 'full_pipeline.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
