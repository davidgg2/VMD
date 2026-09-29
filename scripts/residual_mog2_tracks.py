"""Fuse residual-flow boxes with the bsense MOG2 small-target pipeline.

EdgeTAM is not used. Residual flow keeps camera-motion-compensated regions
from `residual_motion_boxes.py`. MOG2 keeps the high-recall small-blob
detector from `D:\\bsense\\mog_for_small_targets\\mog_morph_tracker.py` and
`fusion_tracker\\vmd_tinycnn_branch.py`. Overlapping boxes are merged and
tracked with the greedy centroid matcher from `mog_morph_tracker_filtered.py`.
MOG2-only tracks are drawn after a trajectory longer than one second.
A track is also drawn as soon as it has both residual and MOG2 evidence
on the same object. Residual-only tracks are not drawn.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


# Residual-flow settings from scripts/residual_motion_boxes.py
FLOW_WIDTH = 320
FLOW_THRESHOLD = 1.0
FLOW_MIN_AREA = 12
FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=21, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)

# MOG2 + morphology from mog_morph_tracker.py / vmd config
MOG2_HISTORY = 300
MOG2_VAR_THR = 12.0
MOG2_FG_THRESH = 200
MORPH_OPEN = 3
MORPH_CLOSE = 5
MORPH_CLOSE_ITERS = 2
MIN_CONTOUR_AREA = 10
MAX_CONTOUR_AREA = 3000
ASPECT_MIN = 0.2
ASPECT_MAX = 5.0

# Fusion + tracker from fusion_tracker/config.yaml and mog_morph_tracker_filtered.py
FUSE_IOU = 0.25
FUSE_CENTROID = 30.0
TRACK_MAX_DISTANCE = 30.0
TRACK_MAX_AGE = 10
# Do not publish MOG2 until the mixture has seen a few frames; frame 0 is
# typically a full-foreground mask. Keep this far below history=300 so short
# clips still get small-target detections.
MOG2_WARMUP = 30
MIN_TRAJECTORY_SECONDS = 1.0

RESIDUAL_COLOR = (40, 220, 70)
MOG2_COLOR = (255, 220, 40)
BOTH_COLOR = (40, 180, 255)
MASK_COLOR = np.array([255, 180, 40], dtype=np.float32)


def union_xyxy(boxes):
    b = np.asarray(boxes, dtype=float)
    return [float(b[:, 0].min()), float(b[:, 1].min()),
            float(b[:, 2].max()), float(b[:, 3].max())]


def center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def excluded_mask(h, w, flow_h, flow_w):
    """Timestamp band and reticle, same geometry as residual_motion_boxes.py."""
    valid = np.ones((h, w), np.uint8)
    valid[:round(h * 14 / 256)] = 0
    y1, y2 = round(h * 120 / 256), round(h * 134 / 256)
    x1, x2 = round(w * 148 / 320), round(w * 163 / 320)
    valid[y1:y2, x1:x2] = 0
    flow_valid = np.ones((flow_h, flow_w), bool)
    flow_valid[:round(flow_h * 14 / 256)] = False
    flow_valid[round(flow_h * 120 / 256):round(flow_h * 134 / 256), 148:163] = False
    return valid, flow_valid


def residual_boxes(prev_gray, gray, w, h, flow_w, flow_h, sx, sy, flow_valid,
                   threshold=FLOW_THRESHOLD, min_area=FLOW_MIN_AREA):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK)
    residual = flow - np.median(flow[flow_valid], axis=0)
    magnitude = np.linalg.norm(residual * [sx, sy], axis=2)
    mask = np.uint8((magnitude > threshold) & flow_valid)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    boxes = []
    for label in range(1, n):
        support = (labels == label) & (mask > 0)
        area = int(support.sum())
        if area < min_area:
            continue
        x, y, bw, bh, _ = stats[label]
        x1, y1 = max(0, int(x * sx) - 3), max(0, int(y * sy) - 3)
        x2, y2 = min(w - 1, int((x + bw) * sx) + 3), min(h - 1, int((y + bh) * sy) + 3)
        boxes.append({
            'xyxy': [x1, y1, x2, y2],
            'source': 'residual',
            'support': area,
            'peak_px_per_frame': float(magnitude[support].max()),
        })
    return boxes


def mog2_boxes(mog2, k_open, k_close, gray_native, valid_native):
    fg = mog2.apply(cv2.GaussianBlur(gray_native, (3, 3), 0))
    _, mask = cv2.threshold(fg, MOG2_FG_THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, valid_native)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=MORPH_CLOSE_ITERS)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / float(bh + 1e-6)
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue
        boxes.append({
            'xyxy': [x, y, x + bw, y + bh],
            'source': 'mog2',
            'support': int(area),
        })
    return boxes, mask


def fuse_boxes(residual, mog2):
    """Union boxes that overlap by IoU or sit within the centroid gate."""
    used_m = set()
    fused = []
    for r in residual:
        matches = []
        for mi, m in enumerate(mog2):
            if mi in used_m:
                continue
            cxr, cyr = center(r['xyxy'])
            cxm, cym = center(m['xyxy'])
            dist = float(np.hypot(cxr - cxm, cyr - cym))
            inside_r = r['xyxy'][0] <= cxm <= r['xyxy'][2] and r['xyxy'][1] <= cym <= r['xyxy'][3]
            inside_m = m['xyxy'][0] <= cxr <= m['xyxy'][2] and m['xyxy'][1] <= cyr <= m['xyxy'][3]
            if iou(r['xyxy'], m['xyxy']) >= FUSE_IOU or dist <= FUSE_CENTROID or inside_r or inside_m:
                matches.append(mi)
        if matches:
            parts = [r['xyxy']] + [mog2[mi]['xyxy'] for mi in matches]
            used_m.update(matches)
            fused.append({
                'xyxy': union_xyxy(parts),
                'source': 'both',
                'support': r['support'] + sum(mog2[mi]['support'] for mi in matches),
            })
        else:
            fused.append(r)
    for mi, m in enumerate(mog2):
        if mi not in used_m:
            fused.append(m)
    return fused


def source_flags(source):
    return source in ('residual', 'both'), source in ('mog2', 'both')


class Track:
    __slots__ = ('id', 'cx', 'cy', 'vx', 'vy', 'bbox', 'source', 'hits', 'missed',
                 'lifetime', 'first_frame', 'seen_residual', 'seen_mog2')

    def __init__(self, tid, det, frame_idx):
        cx, cy = center(det['xyxy'])
        self.id = tid
        self.cx, self.cy = cx, cy
        self.vx = self.vy = 0.0
        self.bbox = det['xyxy']
        self.source = det['source']
        self.hits = 1
        self.missed = 0
        self.lifetime = 1
        self.first_frame = frame_idx
        self.seen_residual, self.seen_mog2 = source_flags(det['source'])

    def predict(self):
        return self.cx + self.vx, self.cy + self.vy

    def update(self, det):
        cx, cy = center(det['xyxy'])
        self.vx, self.vy = cx - self.cx, cy - self.cy
        self.cx, self.cy = cx, cy
        self.bbox = det['xyxy']
        residual, mog2 = source_flags(det['source'])
        self.seen_residual = self.seen_residual or residual
        self.seen_mog2 = self.seen_mog2 or mog2
        self.source = 'both' if self.seen_residual and self.seen_mog2 else det['source']
        self.hits += 1
        self.missed = 0

    def confirmed(self, frame_idx, fps, min_seconds):
        if self.missed != 0 or not self.seen_mog2:
            return False
        if self.seen_residual:
            return True
        span = (frame_idx - self.first_frame + 1) / max(fps, 1e-6)
        return span > min_seconds

    def draw_color(self):
        return BOTH_COLOR if self.seen_residual else MOG2_COLOR


def step_tracker(tracks, dets, next_id, frame_idx):
    if not tracks:
        for det in dets:
            tracks.append(Track(next_id, det, frame_idx))
            next_id += 1
        return next_id
    predicted = [t.predict() for t in tracks]
    pairs = []
    for ti, (px, py) in enumerate(predicted):
        gate = max(TRACK_MAX_DISTANCE, (tracks[ti].bbox[2] - tracks[ti].bbox[0]) * 0.5)
        for di, det in enumerate(dets):
            cx, cy = center(det['xyxy'])
            dist = float(np.hypot(px - cx, py - cy))
            if dist <= gate:
                pairs.append((dist, ti, di))
    pairs.sort()
    used_t, used_d = set(), set()
    for _, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        tracks[ti].update(dets[di])
        used_t.add(ti)
        used_d.add(di)
    for ti, track in enumerate(tracks):
        if ti not in used_t:
            track.missed += 1
    for di, det in enumerate(dets):
        if di not in used_d:
            tracks.append(Track(next_id, det, frame_idx))
            next_id += 1
    for track in tracks:
        track.lifetime += 1
    tracks[:] = [t for t in tracks if t.missed <= TRACK_MAX_AGE]
    return next_id


def draw_legend(frame):
    cv2.rectangle(frame, (8, 10), (20, 22), MOG2_COLOR, -1)
    cv2.putText(frame, 'mog2 > 1s', (24, 22), 0, 0.45, MOG2_COLOR, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (130, 10), (142, 22), BOTH_COLOR, -1)
    cv2.putText(frame, 'residual+mog2', (146, 22), 0, 0.45, BOTH_COLOR, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--mog2-warmup', type=int, default=MOG2_WARMUP)
    p.add_argument('--min-trajectory-seconds', type=float, default=MIN_TRAJECTORY_SECONDS,
                   help='Draw only after this many seconds of trajectory')
    args = p.parse_args()
    video = Path(args.video)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h, reported = [int(cap.get(k)) for k in (3, 4, 7)]
    if min(w, h) <= 0:
        raise RuntimeError('Invalid video metadata')
    flow_w, flow_h = FLOW_WIDTH, round(h * FLOW_WIDTH / w)
    sx, sy = w / flow_w, h / flow_h
    valid_native, flow_valid = excluded_mask(h, w, flow_h, flow_w)

    target = out / f'{video.stem}_residual_mog2_tracks.mp4'
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError('Cannot open video writer')

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THR, detectShadows=False)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN, MORPH_OPEN))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE, MORPH_CLOSE))

    tracks = []
    next_id = 1
    previous = None
    records = []
    samples = []
    sample_ids = set()
    if fps > 0 and reported > 0:
        sample_ids = set(np.linspace(0, max(reported - 1, 0), min(12, max(reported, 1)), dtype=int).tolist())
    start = time.time()
    idx = 0
    frames_with_mog2 = 0
    frames_with_residual = 0
    frames_displayed = 0
    displayed_ids = set()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray_native = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_flow = cv2.resize(gray_native, (flow_w, flow_h))
            r_boxes = []
            if previous is not None:
                r_boxes = residual_boxes(previous, gray_flow, w, h, flow_w, flow_h, sx, sy, flow_valid)
            m_boxes, mog_mask = mog2_boxes(mog2, k_open, k_close, gray_native, valid_native)
            if idx < args.mog2_warmup:
                m_boxes = []
                mog_mask = np.zeros_like(mog_mask)
            fused = fuse_boxes(r_boxes, m_boxes)
            next_id = step_tracker(tracks, fused, next_id, idx)

            live = [t for t in tracks if t.confirmed(idx, fps, args.min_trajectory_seconds)]
            vis = frame.copy()
            overlay = vis.astype(np.float32)
            shown_mask = np.zeros(mog_mask.shape, bool)
            for track in live:
                x1, y1, x2, y2 = map(int, track.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                shown_mask[y1:y2, x1:x2] = mog_mask[y1:y2, x1:x2] > 0
            if shown_mask.any():
                overlay[shown_mask] = overlay[shown_mask] * 0.72 + MASK_COLOR * 0.28
            vis = overlay.astype(np.uint8)
            for track in live:
                x1, y1, x2, y2 = map(int, track.bbox)
                color = track.draw_color()
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f'ID {track.id}', (x1, max(16, y1 - 4)),
                            0, 0.5, color, 1, cv2.LINE_AA)
            draw_legend(vis)
            cv2.putText(vis, f'{idx / fps:.2f}s  res={len(r_boxes)} mog2={len(m_boxes)} tracks={len(live)}',
                        (8, h - 10), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            writer.write(vis)

            if r_boxes:
                frames_with_residual += 1
            if m_boxes:
                frames_with_mog2 += 1
            if live:
                frames_displayed += 1
                displayed_ids.update(t.id for t in live)
            records.append({
                'frame': idx,
                'residual': r_boxes,
                'mog2': [{'xyxy': b['xyxy'], 'support': b['support']} for b in m_boxes],
                'tracks': [{'id': t.id, 'xyxy': t.bbox, 'source': t.source, 'hits': t.hits,
                            'seen_residual': t.seen_residual, 'seen_mog2': t.seen_mog2}
                           for t in live],
            })
            if idx in sample_ids:
                samples.append(cv2.resize(vis, (480, 384)))
            previous = gray_flow
            idx += 1
            if idx % 250 == 0:
                print(f'{idx}/{reported} frames', flush=True)
    finally:
        cap.release()
        writer.release()

    if samples:
        while len(samples) % 3:
            samples.append(np.zeros_like(samples[0]))
        sheet = np.vstack([np.hstack(samples[i:i + 3]) for i in range(0, len(samples), 3)])
        cv2.imwrite(str(out / 'contact_sheet.jpg'), sheet)
    (out / 'detections.json').write_text(json.dumps(records))
    summary = {
        'source': str(video.resolve()),
        'output': str(target.resolve()),
        'frames': idx,
        'fps': fps,
        'size': [w, h],
        'edgetam': False,
        'method': 'Draw MOG2 tracks with trajectory > 1s; also draw immediately when the same track has both residual and MOG2. Residual-only is hidden.',
        'residual': {
            'threshold_native_px_per_frame': FLOW_THRESHOLD,
            'minimum_flow_pixel_area': FLOW_MIN_AREA,
            'flow_size': [flow_w, flow_h],
        },
        'mog2': {
            'history': MOG2_HISTORY,
            'var_threshold': MOG2_VAR_THR,
            'fg_threshold': MOG2_FG_THRESH,
            'min_contour_area': MIN_CONTOUR_AREA,
            'max_contour_area': MAX_CONTOUR_AREA,
            'source': r'D:\bsense\mog_for_small_targets\mog_morph_tracker.py',
        },
        'tracker': {
            'max_distance_px': TRACK_MAX_DISTANCE,
            'max_age': TRACK_MAX_AGE,
            'min_trajectory_seconds': args.min_trajectory_seconds,
            'mog2_only_requires_1s': True,
            'display_immediately_if_residual_and_mog2': True,
            'mog2_warmup_frames': args.mog2_warmup,
        },
        'frames_with_residual': frames_with_residual,
        'frames_with_mog2': frames_with_mog2,
        'frames_with_displayed_tracks': frames_displayed,
        'displayed_track_ids': sorted(displayed_ids),
        'unique_track_ids': next_id - 1,
        'elapsed_seconds': time.time() - start,
    }
    (out / 'settings.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(target, flush=True)


if __name__ == '__main__':
    main()
