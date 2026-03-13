# LeRobot Episode Scorer

Episode-level scoring for LeRobot datasets with quality metrics and optional execution scoring via frozen VLM probes.

## Installation

```bash
uv sync
```

## Quick Start

Score a LeRobot dataset with quality metrics only:

```bash
lerobot-episode-score \
  --repo-id j-m-h/pick_place_clean_realsense_downscaled \
  --output-dir ./outputs/my_scores
```

## CLI Commands

### `lerobot-episode-score`

Score episodes with quality metrics and optional execution backend.

```bash
lerobot-episode-score \
  --repo-id <hf-dataset-id> \
  --output-dir <path> \
  [--root <local-lerobot-root>] \
  [--dataset-family <name>] \
  [--camera-key <key>]... \
  [--nominal-runtime-seconds <seconds>] \
  [--execution-backend none|frozen_vlm_probe] \
  [--execution-checkpoint <path>] \
  [--vlm-model-id <model>] \
  [--device <device>]
```

**Arguments:**
- `--repo-id` (required): HuggingFace dataset repository ID
- `--output-dir` (required): Output directory for scores
- `--root`: Optional local LeRobot data root
- `--dataset-family`: Family name stored in outputs (default: "custom")
- `--camera-key`: Camera keys to score (e.g., `top`, `wrist`). Repeat for multiple cameras
- `--nominal-runtime-seconds`: Target runtime for scoring. Defaults to dataset median
- `--execution-backend`: Scoring backend. `none` (default) or `frozen_vlm_probe`
- `--execution-checkpoint`: Path to checkpoint file (required if backend is `frozen_vlm_probe`)
- `--vlm-model-id`: VLM model ID (default: `ACIDE/FailSense-Calvin-2p-3b`)
- `--device`: Torch device (auto-detected if not specified)

**Outputs:**
- `episode_scores.json` – Full per-episode scores
- `episode_scores.csv` – Flattened CSV export
- `summary.json` – Aggregate metrics

---

### `lerobot-episode-extract-features`

Extract frozen VLM features from episodes for training/evaluation.

```bash
lerobot-episode-extract-features \
  --dataset-manifest <path> \
  --output-dir <path> \
  [--camera-key <key>]... \
  [--frames-per-camera <n>] \
  [--vlm-model-id <model>] \
  [--log-every <n>] \
  [--device <device>]
```

**Arguments:**
- `--dataset-manifest` (required): Path to dataset manifest JSON (see format below)
- `--output-dir` (required): Output directory for feature cache
- `--frames-per-camera`: Frames to sample per camera (default: 4)
- `--vlm-model-id`: VLM for feature extraction (default: `ACIDE/FailSense-Calvin-2p-3b`)
- `--log-every`: Log progress every N episodes (default: 10)

**Outputs:**
- `features/episode_*.pt` – Per-episode feature tensors
- `index.json` – Feature cache index with metadata
- `manifest.json` – Dataset manifest copy
- `summary.json` – Extraction summary
- `progress.json` – Real-time progress tracking

---

### `lerobot-episode-train-execution`

Train an execution classifier on extracted features.

```bash
lerobot-episode-train-execution \
  --feature-dir <path> \
  --output-dir <path> \
  [--batch-size <n>] \
  [--num-epochs <n>] \
  [--learning-rate <lr>] \
  [--weight-decay <wd>] \
  [--hidden-dim <dim>] \
  [--dropout-rate <rate>] \
  [--threshold <threshold>] \
  [--device <device>]
```

**Arguments:**
- `--feature-dir` (required): Directory from `lerobot-episode-extract-features`
- `--output-dir` (required): Output directory for checkpoint
- `--batch-size`: Training batch size (default: 8)
- `--num-epochs`: Training epochs (default: 10)
- `--learning-rate`: AdamW learning rate (default: 1e-4)
- `--weight-decay`: AdamW weight decay (default: 0.01)
- `--hidden-dim`: Per-layer hidden size (default: 512)
- `--dropout-rate`: Head dropout (default: 0.3)
- `--threshold`: Decision threshold stored with checkpoint (default: 0.5)

**Outputs:**
- `checkpoint.pt` – Trained execution checkpoint
- `metrics.json` – Training metrics
- `progress.json` – Training progress

---

### `lerobot-episode-evaluate-execution`

Evaluate a trained checkpoint on cached features.

```bash
lerobot-episode-evaluate-execution \
  --feature-dir <path> \
  --checkpoint <path> \
  --output-dir <path> \
  [--batch-size <n>] \
  [--device <device>]
```

**Arguments:**
- `--feature-dir` (required): Directory from `lerobot-episode-extract-features`
- `--checkpoint` (required): Checkpoint from training
- `--output-dir` (required): Output directory for scores
- `--batch-size`: Evaluation batch size (default: 16)

**Outputs:**
- `episode_scores.json` – Per-episode execution scores
- `episode_scores.csv` – Flattened CSV export
- `summary.json` – Aggregate metrics with baselines

---

## Dataset Manifest Format

Dataset manifests define which episodes to extract and their labels.

```json
{
  "datasets": [
    {
      "repo_id": "j-m-h/pick_place_clean_realsense_downscaled",
      "dataset_family": "clean_reference",
      "split": "train",
      "episode_from": 0,
      "episode_to": 96,
      "derived_label": 1,
      "label_rule": "pure clean episodes => task success",
      "use_for_training": true,
      "use_for_evaluation": false
    },
    {
      "repo_id": "fabiangrob/pick_place_task_fail_realsense_downscaled",
      "dataset_family": "task_fail",
      "split": "val",
      "episode_from": 40,
      "episode_to": 45,
      "derived_label": 0,
      "label_rule": "explicit task-failure dataset => task failure",
      "use_for_training": true,
      "use_for_evaluation": false
    }
  ]
}
```

**Fields:**
- `repo_id`: HuggingFace dataset ID
- `dataset_family`: Grouping name for summaries
- `split`: `train`, `val`, or `test`
- `episode_from` / `episode_to`: Episode range (exclusive end)
- `derived_label`: Force label (1=success, 0=failure, null=use dataset label)
- `label_rule`: Human-readable explanation
- `use_for_training`: Include in training set
- `use_for_evaluation`: Include in evaluation set

---

## Quality Scoring Components

Quality scores (0-1) combine these weighted metrics:

| Component | Weight | Description |
|-----------|--------|-------------|
| `visual_clarity` | 20% | Laplacian variance (blur), contrast, exposure |
| `smoothness` | 10% | Acceleration smoothness |
| `path_efficiency` | 10% | Straight-line vs actual path ratio |
| `collision` | 10% | Acceleration spike detection |
| `joint_stability` | 10% | Final-state joint variance |
| `actuator_saturation` | 10% | Action-state divergence |
| `runtime` | 20% | Deviation from nominal runtime |

---

## Example Workflow

**1. Extract features:**
```bash
lerobot-episode-extract-features \
  --dataset-manifest manifests/pick_place_training.json \
  --output-dir outputs/features_v1 \
  --camera-key top --camera-key wrist
```

**2. Train execution classifier:**
```bash
lerobot-episode-train-execution \
  --feature-dir outputs/features_v1 \
  --output-dir outputs/checkpoint_v1 \
  --num-epochs 15
```

**3. Evaluate on test set:**
```bash
lerobot-episode-evaluate-execution \
  --feature-dir outputs/features_v1 \
  --checkpoint outputs/checkpoint_v1/checkpoint.pt \
  --output-dir outputs/eval_v1
```

**4. Score new dataset with trained checkpoint:**
```bash
lerobot-episode-score \
  --repo-id my-robot/new_episodes \
  --output-dir outputs/new_scores \
  --execution-backend frozen_vlm_probe \
  --execution-checkpoint outputs/checkpoint_v1/checkpoint.pt
```

---

## Testing

```bash
uv run pytest tests/ -v
```

---

## Output Metrics

All binary classification metrics in summaries:

- `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`
- `auroc`, `auprc`
- `brier_score`, `log_loss`
- `prevalence`
