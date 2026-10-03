# Residual + MOG2 + DA-W video pipeline

Offline motion analysis for thermal and visible clips. One entry point runs **three components**. Only component 1 uses neural nets. Components 2 and 3 are classical OpenCV.

## Parameter count

Learned weights come only from the DA-W residual comparison (`analyze_video.py`). Residual boxes and MOG2 add **0** trained parameters.

| Piece | Role | Parameters |
|---|---|---|
| DA-W ViT-S (`daw_vits_stage2.pth`) | Weather-conditioned relative disparity | **24,847,489** |
| DA-W style filter (`daw_style_filter_stage1.pth`) | Style embedding for DA-W | **12,493,536** |
| Depth Anything V2 ViT-S (`depth_anything_v2_vits.pth`) | Unconditioned depth baseline | **24,785,089** |
| Motion encoder Swin-T (`swin_t.pth`) | Dense features (not classes) | **30,985,869** |
| Residual-flow boxes | Farneback + threshold | 0 |
| MOG2 + fused tracker | Background subtraction + greedy tracks | 0 |
| **Full pipeline** | Sum of the four checkpoints | **93,111,983 (~93.1 M)** |

Counts are `numel()` over tensors in each checkpoint under `checkpoints/` (loaded 2026-09-29). DA-W as a pair (depth + style) is **37,341,025**. EdgeTAM is not used.

MOG2 is a per-pixel Gaussian mixture that adapts online. That is not a fixed trained network.

## Three components

```
raw video
    │
    ├─ 1. DA-W residual     → comparison.mp4
    ├─ 2. Residual boxes    → *_daw_residual_boxes.mp4
    └─ 3. Residual + MOG2   → *_three_algorithms.mp4
```

### 1. DA-W residual (`scripts/analyze_video.py`)

Per frame, writes a 6-panel comparison (960×576):

1. Input
2. DA-W relative disparity (not meters)
3. Depth Anything V2 baseline
4. Motion-trained Swin-T features (fixed first-frame PCA; not object IDs)
5. Residual optical flow (Farneback at 320×256, median translation removed)
6. Feature cosine change

These outputs are diagnostics. They are not calibrated depth and not detections.

### 2. Residual boxes (`scripts/residual_boxes_on_comparison.py`)

Same residual as the middle-bottom panel, then:

- threshold **6 native pixels/frame** (tuned so day cars remain and foliage drops)
- minimum **40** above-threshold pixels at 320×256
- 3×3 close and connected components

Green boxes are drawn on the comparison **input** and **residual-flow** tiles.

### 3. Residual + MOG2 (`scripts/three_algorithms_movie.py`)

Side-by-side at native resolution:

| Pane | Algorithm | Display rule |
|---|---|---|
| Left | Residual-flow boxes | Same 6 px / 40-pixel gate |
| Middle | MOG2 VMD | History 300, var 12, open 3×3, close 5×5×2, min contour 10. Tracks shown after **> 1 s** |
| Right | Fused residual + MOG2 | Show immediately if the same track saw both sources; otherwise MOG2-only after 1 s |

MOG2 comes from `D:\bsense\mog_for_small_targets\mog_morph_tracker.py`. Residual-only tracks are not drawn on the fused pane. The first 30 MOG2 frames are held back while the background model warms up.

## Run

From `D:\turbulencedScenario`:

```powershell
& 'D:\TrackAnything\.venv\Scripts\python.exe' -u scripts/full_pipeline.py --video 'PATH\to\movie.mp4'
```

Outputs go to `outputs/full_pipeline/<stem>/`.

| File | Component |
|---|---|
| `comparison.mp4` | 1. DA-W residual |
| `<stem>_daw_residual_boxes.mp4` | 2. Residual boxes on that comparison |
| `<stem>_three_algorithms.mp4` | 3. Residual \| MOG2 \| fused |
| `full_pipeline.json` | Paths and settings |

`--skip-daw` reuses an existing `comparison.mp4`.  
`--threshold-native` and `--min-area` override the 6 / 40 residual gate.

Single-component commands:

```powershell
# 1. DA-W residual only (needs the video-behavior-cuda env)
& 'C:\Users\DavidGlickman\.venvs\video-behavior-cuda\Scripts\python.exe' -u scripts/analyze_video.py --video 'PATH\to\movie.mp4' --output outputs/analysis/<stem>

# 2. Residual boxes on a comparison movie
& 'D:\TrackAnything\.venv\Scripts\python.exe' -u scripts/residual_boxes_on_comparison.py --comparison outputs/analysis/<stem>/comparison.mp4 --source 'PATH\to\movie.mp4' --output outputs/analysis/<stem>/comparison_residual_boxes.mp4

# 3. Residual + MOG2 side-by-side
& 'D:\TrackAnything\.venv\Scripts\python.exe' -u scripts/three_algorithms_movie.py --videos 'PATH\to\movie.mp4'
```

Batch residual+MOG2 (no DA-W) over the raw movie folders: `scripts/batch_residual_mog2.py`.

## Environments

- **DA-W residual:** `C:\Users\DavidGlickman\.venvs\video-behavior-cuda\Scripts\python.exe` (torch, CUDA)
- **Residual boxes and MOG2:** `D:\TrackAnything\.venv\Scripts\python.exe` (OpenCV)

Weights stay in `checkpoints/` (not committed). Movies are ignored by `.gitignore`.

## Provenance

- [DA-W](https://github.com/taco-group/DA-W) ViT-S + style filter
- [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)
- [Object Concepts Emerge from Motion](https://github.com/TJ12342/object-concepts-from-motion) Swin-T
- MOG2 morphology: `D:\bsense\mog_for_small_targets\mog_morph_tracker.py`

Relative disparity is not metric depth. Residual flow mixes real motion, turbulence, noise, and leftover camera motion. Feature colors are not class labels.
