# LeRobot Episode Scorer

Episode-level scoring for LeRobot datasets with quality metrics and VLM-based execution scoring.

## Installation

```bash
uv sync
```

For the default VLM backend, start the LM Studio local server with a vision-capable model loaded.
The scorer defaults to `qwen/qwen3.5-9b` on `http://localhost:1234/v1`.

Optional: Ollama is still supported as an alternate backend.

## Quick Start

Score a LeRobot dataset with the default LM Studio backend:

```bash
lerobot-episode-score \
  --repo-id j-m-h/pick_place_clean_realsense_downscaled \
  --output-dir ./outputs/my_scores
```

Score with quality metrics only:

```bash
lerobot-episode-score \
  --repo-id j-m-h/pick_place_clean_realsense_downscaled \
  --output-dir ./outputs/my_scores \
  --execution-backend none
```

## CLI Commands

### `lerobot-episode-score`

Score episodes with quality metrics and optional VLM-based execution scoring.

```bash
lerobot-episode-score \
  --repo-id <hf-dataset-id> \
  --output-dir <path> \
  [--root <local-lerobot-root>] \
  [--dataset-family <name>] \
  [--camera-key <key>]... \
  [--nominal-runtime-seconds <seconds>] \
  [--execution-backend none|lmstudio|ollama] \
  [--lmstudio-model <model>] \
  [--lmstudio-base-url <url>] \
  [--ollama-model <model>] \
  [--ollama-host <url>] \
  [--stitch-border-size <pixels>] \
  [--think]
```

**Arguments:**
- `--repo-id` (required): HuggingFace dataset repository ID
- `--output-dir` (required): Output directory for scores
- `--root`: Optional local LeRobot data root
- `--dataset-family`: Family name stored in outputs (default: `custom`)
- `--camera-key`: Camera keys to score (e.g. `top`, `wrist`). Repeat for multiple cameras
- `--nominal-runtime-seconds`: Target runtime for scoring. Defaults to dataset median
- `--execution-backend`: Scoring backend. `lmstudio` by default, or `none` / `ollama`
- `--lmstudio-model`: LM Studio model to use (default: `qwen/qwen3.5-9b`)
- `--lmstudio-base-url`: LM Studio OpenAI-compatible base URL (default: `http://localhost:1234/v1`)
- `--ollama-model`: Ollama model to use (default: `qwen3.5:0.8b`)
- `--ollama-host`: Ollama host URL (default: `http://localhost:11434`)
- `--stitch-border-size`: Border size in pixels for stitched frame grid (default: `4`)
- `--think`: Enable reasoning traces. Disabled by default

**Outputs:**
- `episode_scores.json` - Full per-episode scores
- `episode_scores.csv` - Flattened CSV export
- `summary.json` - Aggregate metrics

## How It Works

### Quality Scoring

Quality scores (0-1) are computed from robot state/action data:

| Component | Weight | Description |
|-----------|--------|-------------|
| `visual_clarity` | 20% | Laplacian variance (blur), contrast, exposure |
| `smoothness` | 10% | Acceleration smoothness |
| `path_efficiency` | 10% | Straight-line vs actual path ratio |
| `collision` | 10% | Acceleration spike detection |
| `joint_stability` | 10% | Final-state joint variance |
| `actuator_saturation` | 10% | Action-state divergence |
| `runtime` | 20% | Deviation from nominal runtime |

### VLM Execution Scoring

When using a VLM backend:

1. **Frame Sampling**: 4 frames are sampled from the episode video at strategic positions.
2. **Image Stitching**: Frames are arranged in a 2x2 grid.
3. **Image Preparation**: The stitched image is downscaled before upload to reduce latency.
4. **VLM Query**: The image and task description are sent to the local inference server.
5. **Structured Output**: LM Studio returns JSON so `vlm_response` and `reasoning_trace` stay separate when reasoning is enabled.
6. **Final Score**: Combined as `quality_score * execution_score`.

## Example Workflow

**Default LM Studio backend:**

```bash
lerobot-episode-score \
  --repo-id my-robot/dataset \
  --output-dir ./outputs/vlm_scored
```

**Enable reasoning traces:**

```bash
lerobot-episode-score \
  --repo-id my-robot/dataset \
  --output-dir ./outputs/vlm_scored \
  --think
```

**Use Ollama instead:**

```bash
lerobot-episode-score \
  --repo-id my-robot/dataset \
  --output-dir ./outputs/vlm_scored \
  --execution-backend ollama
```

## Testing

```bash
uv run pytest tests/ -v
```

## Output Metrics

All binary classification metrics in summaries when labels are available:

- `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`
- `auroc`, `auprc`
- `brier_score`, `log_loss`
- `prevalence`
