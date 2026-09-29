"""Render residual, MOG2, and fused residual+MOG2 side by side.

Pane 1 — residual-flow boxes (same threshold analysis as
`residual_boxes_on_comparison.py` / `analyze_video.py`).
Pane 2 — MOG2 tracks longer than 1 second.
Pane 3 — fused residual+MOG2 (MOG2>1s, or immediate if both sources).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_mog2_tracks as rmt


def label_pane(frame, title, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(frame, title, (8, 20), 0, 0.55, color, 1, cv2.LINE_AA)
    return frame


def draw_xyxy(frame, box, color, text=None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(frame, text, (x1, max(40, y1 - 4)), 0, 0.45, color, 1, cv2.LINE_AA)


def process_video(video: Path, dest: Path, mog2_warmup: int, min_seconds: float,
                  residual_threshold: float, residual_min_area: int) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h, reported = [int(cap.get(k)) for k in (3, 4, 7)]
    flow_w, flow_h = rmt.FLOW_WIDTH, round(h * rmt.FLOW_WIDTH / w)
    sx, sy = w / flow_w, h / flow_h
    valid_native, flow_valid = rmt.excluded_mask(h, w, flow_h, flow_w)

    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 3, h))
    if not writer.isOpened():
        raise RuntimeError('Cannot open video writer')

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=rmt.MOG2_HISTORY, varThreshold=rmt.MOG2_VAR_THR, detectShadows=False)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rmt.MORPH_OPEN, rmt.MORPH_OPEN))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rmt.MORPH_CLOSE, rmt.MORPH_CLOSE))

    mog2_tracks, fused_tracks = [], []
    mog2_id = fused_id = 1
    previous = None
    start = time.time()
    idx = 0
    counts = {'residual': 0, 'mog2': 0, 'fused': 0}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray_native = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_flow = cv2.resize(gray_native, (flow_w, flow_h))
            r_boxes = []
            if previous is not None:
                r_boxes = rmt.residual_boxes(
                    previous, gray_flow, w, h, flow_w, flow_h, sx, sy, flow_valid,
                    threshold=residual_threshold, min_area=residual_min_area)
            m_boxes, mog_mask = rmt.mog2_boxes(mog2, k_open, k_close, gray_native, valid_native)
            if idx < mog2_warmup:
                m_boxes = []
                mog_mask = np.zeros_like(mog_mask)

            mog2_id = rmt.step_tracker(mog2_tracks, m_boxes, mog2_id, idx)
            fused = rmt.fuse_boxes(r_boxes, m_boxes)
            fused_id = rmt.step_tracker(fused_tracks, fused, fused_id, idx)

            mog2_live = [t for t in mog2_tracks
                         if t.missed == 0 and t.seen_mog2
                         and (idx - t.first_frame + 1) / max(fps, 1e-6) > min_seconds]
            fused_live = [t for t in fused_tracks if t.confirmed(idx, fps, min_seconds)]

            residual_pane = frame.copy()
            mog2_pane = frame.copy()
            fused_pane = frame.copy()
            overlay = fused_pane.astype(np.float32)
            shown = np.zeros(mog_mask.shape, bool)
            for track in fused_live:
                x1, y1, x2, y2 = map(int, track.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                shown[y1:y2, x1:x2] = mog_mask[y1:y2, x1:x2] > 0
            if shown.any():
                overlay[shown] = overlay[shown] * 0.72 + rmt.MASK_COLOR * 0.28
            fused_pane = overlay.astype(np.uint8)

            for box in r_boxes:
                draw_xyxy(residual_pane, box['xyxy'], rmt.RESIDUAL_COLOR)
            for track in mog2_live:
                draw_xyxy(mog2_pane, track.bbox, rmt.MOG2_COLOR, f'ID {track.id}')
            for track in fused_live:
                draw_xyxy(fused_pane, track.bbox, track.draw_color(), f'ID {track.id}')

            residual_pane = label_pane(
                residual_pane,
                f'1 residual flow thr={residual_threshold:g}px  boxes={len(r_boxes)}',
                rmt.RESIDUAL_COLOR)
            mog2_pane = label_pane(mog2_pane, f'2 mog2 > {min_seconds:g}s  tracks={len(mog2_live)}', rmt.MOG2_COLOR)
            fused_pane = label_pane(fused_pane, f'3 residual+mog2  tracks={len(fused_live)}', rmt.BOTH_COLOR)
            writer.write(np.hstack([residual_pane, mog2_pane, fused_pane]))

            if r_boxes:
                counts['residual'] += 1
            if mog2_live:
                counts['mog2'] += 1
            if fused_live:
                counts['fused'] += 1
            previous = gray_flow
            idx += 1
            if idx % 250 == 0:
                print(f'  {idx}/{reported} {video.name}', flush=True)
    finally:
        cap.release()
        writer.release()

    return {
        'source': str(video.resolve()),
        'output': str(dest.resolve()),
        'frames': idx,
        'fps': fps,
        'size': [w, h],
        'frames_with_residual_boxes': counts['residual'],
        'frames_with_mog2_tracks': counts['mog2'],
        'frames_with_fused_tracks': counts['fused'],
        'residual_threshold_native_px': residual_threshold,
        'residual_min_area': residual_min_area,
        'elapsed_seconds': time.time() - start,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos', nargs='+', required=True)
    p.add_argument('--output', default=str(Path(__file__).resolve().parents[1]
                                           / 'outputs' / 'residual_mog2_masked_videos'
                                           / 'three_algorithms'))
    p.add_argument('--threshold-native', type=float, default=6.0)
    p.add_argument('--min-area', type=int, default=40)
    args = p.parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in args.videos:
        video = Path(path)
        dest = out_root / f'{video.stem}_three_algorithms.mp4'
        print(f'Processing {video}', flush=True)
        summary = process_video(
            video, dest, rmt.MOG2_WARMUP, rmt.MIN_TRAJECTORY_SECONDS,
            args.threshold_native, args.min_area)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    (out_root / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print(out_root, flush=True)


if __name__ == '__main__':
    main()
