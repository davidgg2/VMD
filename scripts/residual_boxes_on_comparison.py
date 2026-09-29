"""Draw residual-flow boxes on the DA-W comparison movie.

Uses the same Farneback residual as `analyze_video.py` (median translation
removed at 320x256). Boxes are the connected components of a thresholded
residual, drawn on the input panel and the middle-bottom residual panel.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

PANEL_W, PANEL_H, BAR = 320, 256, 32
# Residual panel is the center tile of the bottom row.
FLOW_X, FLOW_Y = PANEL_W, BAR + PANEL_H + BAR
INPUT_X, INPUT_Y = 0, BAR
BOX_COLOR = (0, 255, 0)


def residual_mask(prev_gray, gray, valid, threshold_flow):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, .5, 3, 21, 3, 5, 1.2, 0)
    residual = np.linalg.norm(flow - np.median(flow[valid], axis=0), axis=2)
    mask = np.uint8((residual > threshold_flow) & valid)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return residual, closed, mask


def boxes_from_mask(closed, raw, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    boxes = []
    for label in range(1, n):
        support = (labels == label) & (raw > 0)
        area = int(support.sum())
        if area < min_area:
            continue
        x, y, bw, bh, _ = stats[label]
        boxes.append((x, y, x + bw, y + bh, area))
    return boxes


def draw_boxes(frame, boxes, ox, oy):
    for x1, y1, x2, y2, _ in boxes:
        cv2.rectangle(frame, (ox + x1, oy + y1), (ox + x2, oy + y2), BOX_COLOR, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', required=True)
    p.add_argument('--source', required=True,
                   help='Original video used to recompute residual (same as analyze_video.py)')
    p.add_argument('--output', required=True)
    p.add_argument('--threshold-native', type=float, default=6.0,
                   help='Residual threshold in native pixels per frame (6 kept cars on day_raw)')
    p.add_argument('--min-area', type=int, default=40)
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    comp = cv2.VideoCapture(args.comparison)
    src = cv2.VideoCapture(args.source)
    if not comp.isOpened() or not src.isOpened():
        raise RuntimeError('Cannot open comparison or source')
    fps = comp.get(cv2.CAP_PROP_FPS) or 25.0
    cw, ch = int(comp.get(3)), int(comp.get(4))
    sw, sh = int(src.get(3)), int(src.get(4))
    total = int(comp.get(7))
    if (cw, ch) != (960, 576):
        print(f'Warning: expected 960x576 comparison, got {cw}x{ch}', flush=True)

    sx = sw / PANEL_W
    threshold_flow = args.threshold_native / max(sx, 1e-6)
    valid = np.ones((PANEL_H, PANEL_W), bool)
    valid[:14] = False
    valid[120:134, 148:163] = False

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (cw, ch))
    if not writer.isOpened():
        raise RuntimeError('Cannot open video writer')

    prev = None
    samples = []
    sample_ids = set(np.linspace(0, max(total - 1, 0), min(12, max(total, 1)), dtype=int).tolist())
    frames_with_boxes = 0
    start = time.time()
    idx = 0
    try:
        while True:
            ok_c, canvas = comp.read()
            ok_s, frame = src.read()
            if not ok_c or not ok_s:
                break
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (PANEL_W, PANEL_H))
            boxes = []
            if prev is not None:
                _, closed, raw = residual_mask(prev, gray, valid, threshold_flow)
                boxes = boxes_from_mask(closed, raw, args.min_area)
            vis = canvas.copy()
            draw_boxes(vis, boxes, INPUT_X, INPUT_Y)
            draw_boxes(vis, boxes, FLOW_X, FLOW_Y)
            cv2.putText(vis, f'residual boxes={len(boxes)}  thr={args.threshold_native:g} native px',
                        (8, ch - 10), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            writer.write(vis)
            if boxes:
                frames_with_boxes += 1
            if idx in sample_ids:
                samples.append(vis.copy())
            prev = gray
            idx += 1
            if idx % 250 == 0:
                print(f'{idx}/{total} frames', flush=True)
    finally:
        comp.release()
        src.release()
        writer.release()

    if samples:
        thumb = [cv2.resize(im, (480, 288)) for im in samples]
        while len(thumb) % 3:
            thumb.append(np.zeros_like(thumb[0]))
        sheet = np.vstack([np.hstack(thumb[i:i + 3]) for i in range(0, len(thumb), 3)])
        cv2.imwrite(str(out.with_name(out.stem + '_contact_sheet.jpg')), sheet)
    summary = {
        'comparison': str(Path(args.comparison).resolve()),
        'source': str(Path(args.source).resolve()),
        'output': str(out.resolve()),
        'frames': idx,
        'fps': fps,
        'threshold_native_px_per_frame': args.threshold_native,
        'threshold_flow_px': threshold_flow,
        'minimum_flow_pixel_area': args.min_area,
        'frames_with_boxes': frames_with_boxes,
        'elapsed_seconds': time.time() - start,
        'method': 'Farneback residual as in analyze_video.py; threshold; 3x3 close; connected components; boxes drawn on comparison input and residual-flow panels.',
    }
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(out, flush=True)


if __name__ == '__main__':
    main()
