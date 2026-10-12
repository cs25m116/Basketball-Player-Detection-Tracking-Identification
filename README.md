# 🏀 Basketball AI: Detect, Track and Identify Players

An end-to-end computer vision pipeline that turns raw basketball broadcast footage into structured analytics: **who is on the court, where they are, and how much of the court each team controls.**

The pipeline chains detection, segmentation-based tracking, unsupervised team clustering, jersey-number OCR, and court homography. Each stage feeds the next.

> Based on the Roboflow *Basketball AI* notebook and the [`roboflow/sports`](https://github.com/roboflow/sports) library (`feat/basketball` branch). The demo footage is from Game 1 of the 2025 NBA Playoffs, Boston Celtics vs New York Knicks.

---

## Table of Contents

1. [Why this is hard](#why-this-is-hard)
2. [Pipeline overview](#pipeline-overview)
3. [Models used](#models-used)
4. [Stage-by-stage walkthrough](#stage-by-stage-walkthrough)
5. [Dataset](#dataset)
6. [Getting started](#getting-started)
7. [Notebook structure](#notebook-structure)
8. [Results and observations](#results-and-observations)
9. [Limitations and future work](#limitations-and-future-work)
10. [Acknowledgements](#acknowledgements)

---

## Why this is hard

Basketball broadcast video is a difficult setting for off-the-shelf vision models:

- **Fast game pace.** Players and the ball move quickly, with heavy motion blur.
- **Constant interaction and overlap.** Players occlude each other all the time.
- **Small, often hidden jersey numbers.** Numbers are tiny, distorted by pose and camera angle, and frequently occluded.
- **Low contrast backgrounds.** Jersey and skin colours can be close to the court and crowd colours.
- **A moving camera.** Different parts of the court are visible from frame to frame, so pixel positions are not stable.

---

## Pipeline overview

```
Video frame
   │
   ▼
┌──────────────────────┐
│ 1. Detection         │  RF-DETR (DINOv2 backbone)
│    players, ball,    │  → boxes for 10 classes
│    numbers, rim…     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. Tracking          │  SAM 2, prompted with RF-DETR boxes
│    pixel-level masks │  → stable tracker IDs through occlusion
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. Team separation   │  SigLIP embeddings → UMAP → K-Means (k=2)
│    (unsupervised)    │  → team ID per tracker
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. Jersey OCR        │  Fine-tuned SmolVLM2 on number crops
│                      │  → number paired with tracked player
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. Player identity   │  Multi-frame number validation + roster lookup
│                      │  → names, numbers, team colours
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 6. Homography        │  33 court keypoints → 2D top-down court view
│                      │  → positions, paths, shot events
└──────────────────────┘
```

---

## Models used

| Stage | Model | Role |
|---|---|---|
| Detection | **RF-DETR Small** (DINOv2-S backbone, a self-supervised ViT) | Robust detection of players, ball, numbers, rim and actions. The DINOv2 backbone generalises quickly to new domains such as basketball. |
| Tracking | **SAM 2.1 Hiera-Large** (~220M parameters) | Pixel-level segmentation and tracking. It is the slowest component but gives the best quality masks. |
| Team clustering | **SigLIP** + **UMAP** + **K-Means** | Label-free team assignment from jersey appearance. |
| Jersey OCR | **SmolVLM2** (fine-tuned) | Reads jersey numbers from small crops. |
| Court mapping | **Keypoint detector** (33 court keypoints) + homography | Maps the video frame to a standard 2D court layout. |

---

## Stage-by-stage walkthrough

### 1. Object detection (RF-DETR)

A fine-tuned RF-DETR model detects 10 classes: `ball`, `ball-in-basket`, `number`, `player`, `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `referee` and `rim`.

- Confidence threshold: `0.4`
- The detections seed both the tracker (player boxes) and the OCR stage (number boxes).

### 2. Player tracking (SAM 2)

**Why not SORT or DeepSORT/ByteTrack?**

| Approach | How it works | Weakness here |
|---|---|---|
| **SORT** | Predicts each box in the next frame with a Kalman filter and matches new detections by IoU overlap. | Motion models break under fast movement and overlap. |
| **DeepSORT / ByteTrack** | Adds appearance embeddings from crops for visual similarity. | Players move fast and overlap often, so crop embeddings become unreliable. |
| **SAM 2 (used here)** | Does not depend on predicted motion or detection matching. It follows how each object looks at pixel level (shape, colour, texture) across frames and stores this as high-dimensional visual embeddings. | Slowest component, but it handles occlusion, deformation and re-entry well. |

**How it is used:** the RF-DETR player boxes from the first frame prompt SAM 2, which then propagates a mask and a stable `tracker_id` for each player through the clip.

**Cleanup step.** SAM 2 sometimes produces masks with disconnected fragments. These errors look harmless alone but **cascade** into later stages such as the clustering crops. A cleanup function drops disconnected parts that lie too far from the main mask, using a centroid and distance check.

### 3. Team separation (unsupervised)

A supervised classifier would need large labelled data, and jerseys are not fixed identities. A new jersey colour would force retraining. So team assignment is unsupervised:

1. Sample frames (stride of 30) and detect players with RF-DETR.
2. Crop each detected player, using the central region to emphasise jersey colour and texture.
3. Embed the crops with **SigLIP**.
4. Reduce the embeddings to 3D with **UMAP**.
5. Cluster with **K-Means (k = 2)**.

The output is two clusters, one per team. Cluster IDs are arbitrary, so a `TEAM_NAMES` mapping (for example `0 → New York Knicks`) is chosen after inspecting the result. A VLM such as Qwen 3 VL could name the teams automatically, but it would add latency, so a manual mapping is used instead.

Team IDs are assigned **once per tracker** and reused across frames.

### 4. Jersey number recognition (OCR)

Numbers are small, partly occluded, distorted by pose, blurred by motion and look different from different camera angles. Pose matters too: for example, `10` can read as `0` when the player is sideways.

- **Zero-shot SmolVLM2:** about **56%** accuracy.
- **Fine-tuned SmolVLM2:** about **86%** accuracy.

Fine-tuning data was built by:

1. Filtering detections to the `number` class only.
2. Resizing crops to **224 × 224** with fill.
3. Auto-annotating them with a VLM, tracking a player with SAM 2 across many frames to collect enough samples, then checking the labels manually.

### 5. Patching results from different models

RF-DETR detects the number box, SmolVLM2 reads it, and SAM 2 provides the player mask with tracker and team IDs. These outputs come from separate models and have to be merged.

- **Matching rule: IoS (Intersection over Smaller area).** A number box lies inside its player's mask, so IoS ≈ 1. IoS is like IoU but normalised by the smaller area, which suits a small box inside a large mask.
- **Multi-frame confirmation.** A single frame often hides the number, so a `ConsecutiveValueTracker` (`n_consecutive = 3`) only locks a number to a tracker after repeated consistent reads.
- **Roster lookup.** Once confirmed, a dictionary maps `(team, number)` to the player's name.

### 6. Homography and court mapping

The goal is to go from *who is on court* to *where they are* to *how much of the court each team controls*.

- A keypoint model detects up to **33 court landmarks** (confidence `0.3`, anchor confidence `0.5`).
- A homography matrix maps the 2D video frame to a standard 2D court layout, giving a top-down view for analysis.
- Because the camera moves constantly, the visible keypoints change from frame to frame. The transform is recomputed per frame.
- **Path cleaning.** Teleport-like jumps are removed with robust speed analysis, gaps are filled by linear interpolation, and paths are smoothed with a **Savitzky–Golay filter**.
- **Shot events.** `ShotEventTracker` combines jump-shot, layup/dunk and ball-in-basket detections to classify shots as made or missed, and plots them on the court.

---

## Dataset

| Property | Value |
|---|---|
| Classes | 10 |
| Source resolution | 640 × 640 |
| Model input | 512 × 512 (resize with fill, because the model expects a square input) |
| Number-crop size (OCR) | 224 × 224 |
| Demo footage | Celtics vs Knicks, 2025 NBA Playoffs Game 1 (10 sample clips) |

---

## Getting started

### Requirements

- Python 3.10+
- A CUDA GPU (a T4 or L4 on Colab is enough)
- A [Roboflow API key](https://app.roboflow.com/settings/api), used to pull the RF-DETR detector, the SmolVLM2 number recogniser and the court keypoint model
- A [HuggingFace token](https://huggingface.co/settings), used to pull SigLIP

### Setup

```bash
# SAM 2 (real-time fork)
git clone https://github.com/Gy920/segment-anything-2-real-time.git
cd segment-anything-2-real-time
pip install -e .
python setup.py build_ext --inplace
(cd checkpoints && bash download_ckpts.sh)
cd ..

# Dependencies
pip install gdown inference-gpu supervision==0.27.0 transformers num2words
pip install git+https://github.com/roboflow/sports.git@feat/basketball
```

### API keys

```python
import os
os.environ["HF_TOKEN"] = "<your-huggingface-token>"
os.environ["ROBOFLOW_API_KEY"] = "<your-roboflow-key>"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
```

On Colab, store both keys under **Secrets** as `HF_TOKEN` and `ROBOFLOW_API_KEY` and read them with `google.colab.userdata`.

### Run

Open `basketball_ai_how_to_detect_track_and_identify_basketball_players.ipynb` and run the cells top to bottom. Sample videos are downloaded automatically with `gdown`.

---

## Notebook structure

| Section | What it does |
|---|---|
| Environment setup | API keys, GPU check, SAM 2 and dependency install |
| Source videos and rosters | Loads sample clips and the `TEAM_ROSTERS` / `TEAM_COLORS` dictionaries |
| Object detection | RF-DETR on a single frame, filtered classes, then the full video |
| Player tracking | SAM 2 prompted by RF-DETR boxes, propagated across the clip |
| Cluster players into teams | SigLIP → UMAP → K-Means, then full-video team assignment |
| Player numbers OCR | SmolVLM2 reads number crops, with IoS matching and consecutive-agreement validation |
| Player recognition | Overlays names, numbers, team colours and masks |
| Court keypoints detection | Loads the keypoint model and keeps high-confidence points |
| Map player positions to court | Homography to a top-down court, then path cleaning |
| Classify shots as made or miss | Jump-shot detection plus shot-event tracking |

---

## Results and observations

- **Detection:** the DINOv2-based RF-DETR adapts quickly to the basketball domain.
- **Tracking:** SAM 2 keeps IDs stable through overlap and brief disappearances, at the cost of speed.
- **Teams:** the unsupervised pipeline separates the two teams without labels and without retraining for new kits.
- **OCR:** fine-tuning lifts jersey recognition from about 56% to about 86%, and multi-frame confirmation makes the final assignment more reliable than any single read.
- **Court view:** homography turns per-frame detections into a continuous top-down view with smooth player paths.

---

## Limitations and future work

- **Cascading errors.** Segmentation mistakes propagate to clustering and OCR, so cleanup steps are essential.
- **Speed.** SAM 2 Large is the bottleneck. A smaller variant trades some quality for speed.
- **OCR in difficult cases.** Heavy motion blur, occlusion and side views remain the main failure modes.
- **Manual team naming.** Cluster-to-team mapping is manual; an open-source VLM could automate it at a latency cost.
- **Single-camera input.** Results depend on keypoint visibility in broadcast footage.

---

## Acknowledgements

- [Roboflow Notebooks](https://github.com/roboflow/notebooks) and [`roboflow/sports`](https://github.com/roboflow/sports) for the original notebook and utilities
- [Meta SAM 2](https://github.com/facebookresearch/sam2) and the [real-time fork](https://github.com/Gy920/segment-anything-2-real-time)
- RF-DETR, DINOv2, SigLIP, SmolVLM2 and UMAP authors
