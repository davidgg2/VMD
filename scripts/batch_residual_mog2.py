"""Batch residual-flow + MOG2 tracking on every raw source MP4.

Uses the current display rules: MOG2 tracks longer than 1s, plus immediate
display when residual and MOG2 sit on the same track. EdgeTAM is not used.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r'D:\TrackAnything\.venv\Scripts\python.exe')
SOURCE_ROOTS = [
    Path(r'D:\gg_cam\movies\videos_for_shalom'),
    Path(r'D:\bsense\movies_from_bsense'),
    Path(r'D:\bsense\movies'),
    Path(r'D:\bsense\movies180526'),
]
DEST = ROOT / 'outputs' / 'residual_mog2_masked_videos'
RUNS = ROOT / 'outputs' / 'residual_mog2_runs'
MANIFEST = DEST / 'manifest.json'


def list_raw_videos():
    videos = []
    skipped = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            skipped.append({'path': str(root), 'reason': 'missing_folder'})
            continue
        for path in sorted(root.glob('*.mp4')):
            if path.stat().st_size <= 0:
                skipped.append({'path': str(path), 'reason': 'empty_file'})
                continue
            videos.append(path)
    return videos, skipped


def work_dir(video: Path) -> Path:
    return RUNS / video.parent.name / video.stem


def dest_path(video: Path) -> Path:
    folder = DEST / video.parent.name
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f'{video.stem}_residual_mog2_tracks.mp4'


def run_step(extra: list[str], log_file: Path) -> None:
    cmd = [str(PYTHON), '-u', str(ROOT / 'scripts' / 'residual_mog2_tracks.py'), *extra]
    with log_file.open('a', encoding='utf-8') as log:
        log.write(f'\n$ {" ".join(cmd)}\n')
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'residual_mog2_tracks.py failed with exit code {proc.returncode}')


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding='utf-8'))
    return {'videos': {}, 'skipped': []}


def save_manifest(manifest: dict) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def process_video(video: Path) -> dict:
    out = work_dir(video)
    dest = dest_path(video)
    out.mkdir(parents=True, exist_ok=True)
    log_file = out / 'batch.log'
    record = {
        'source': str(video),
        'work_dir': str(out),
        'output': str(dest),
        'status': 'running',
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    if dest.exists() and dest.stat().st_size > 0 and (out / 'settings.json').exists():
        record.update(status='skipped_existing', finished=time.strftime('%Y-%m-%dT%H:%M:%S'))
        return record
    start = time.time()
    try:
        run_step(['--video', str(video), '--output', str(out)], log_file)
        produced = out / f'{video.stem}_residual_mog2_tracks.mp4'
        if not produced.exists():
            raise RuntimeError(f'Final movie missing: {produced}')
        shutil.copy2(produced, dest)
        summary = json.loads((out / 'settings.json').read_text(encoding='utf-8'))
        record.update({
            'status': 'ok',
            'frames': summary.get('frames'),
            'fps': summary.get('fps'),
            'size': summary.get('size'),
            'displayed_track_ids': summary.get('displayed_track_ids'),
            'frames_with_displayed_tracks': summary.get('frames_with_displayed_tracks'),
            'elapsed_seconds': time.time() - start,
        })
    except Exception as exc:
        record.update({
            'status': 'error',
            'error': str(exc),
            'traceback': traceback.format_exc(),
            'elapsed_seconds': time.time() - start,
        })
    record['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    return record


def main():
    if not PYTHON.exists():
        raise SystemExit(f'Missing interpreter: {PYTHON}')
    DEST.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    videos, skipped = list_raw_videos()
    manifest = load_manifest()
    manifest['skipped'] = skipped
    manifest['source_roots'] = [str(p) for p in SOURCE_ROOTS]
    manifest['destination'] = str(DEST)
    manifest['total'] = len(videos)
    save_manifest(manifest)
    print(f'Found {len(videos)} raw MP4s; skipped {len(skipped)} empty/missing', flush=True)
    videos.sort(key=lambda p: p.stat().st_size)
    for i, video in enumerate(videos, 1):
        print(f'[{i}/{len(videos)}] {video}', flush=True)
        record = process_video(video)
        manifest['videos'][str(video)] = record
        save_manifest(manifest)
        print(f"  {record['status']} -> {record.get('output')}", flush=True)
    ok = sum(1 for r in manifest['videos'].values() if r.get('status') in {'ok', 'skipped_existing'})
    err = sum(1 for r in manifest['videos'].values() if r.get('status') == 'error')
    print(json.dumps({'completed': ok, 'errors': err, 'total': len(videos), 'destination': str(DEST)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
