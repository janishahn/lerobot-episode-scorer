from collections.abc import Callable
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from lerobot_episode_scorer.dataset import EpisodeRecord
from lerobot_episode_scorer.metrics import compute_binary_metrics, sanitized_metrics
from lerobot_episode_scorer.video import sample_segment_frames

BASE_MODEL_ID = "google/paligemma2-3b-mix-224"
DEFAULT_VLM_MODEL_ID = "ACIDE/FailSense-Calvin-2p-3b"
DEFAULT_FRAMES_PER_CAMERA = 4
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class DatasetManifestEntry:
    repo_id: str
    dataset_family: str
    split: str
    root: Path | None
    episode_from: int
    episode_to: int | None
    derived_label: int | None
    label_rule: str | None
    use_for_training: bool
    use_for_evaluation: bool


@dataclass(frozen=True)
class FeatureIndexEntry:
    feature_path: str
    repo_id: str
    dataset_family: str
    split: str
    use_for_training: bool
    use_for_evaluation: bool
    episode_index: int
    task: str
    label: int | None
    quality_score: float
    decoded_vote: str | None
    num_tokens: int


@dataclass(frozen=True)
class FeatureCacheIndex:
    version: int
    vlm_model_id: str
    camera_keys: list[str]
    frames_per_camera: int
    hook_layer_indices: list[int]
    feature_dim: int
    records: list[FeatureIndexEntry]


@dataclass(frozen=True)
class ExecutionResult:
    score: float
    probability: float
    decoded_vote: str | None


@dataclass(frozen=True)
class ExecutionCheckpoint:
    checkpoint_path: Path
    backend_name: str
    backend_version: int
    vlm_model_id: str
    camera_keys: list[str]
    frames_per_camera: int
    hook_layer_indices: list[int]
    feature_dim: int
    hidden_dim: int
    dropout_rate: float
    threshold: float
    calibration_scale: float
    calibration_bias: float
    label_mapping: dict[str, int]
    head_state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class FeatureBatch:
    layer_features: list[torch.Tensor]
    mask: torch.Tensor
    labels: torch.Tensor
    entries: list[FeatureIndexEntry]


def build_prompt(images: list[Image.Image], task: str) -> str:
    return " ".join(["<image>"] * len(images)) + " evaluate en " + task


def normalize_label(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "success", "pass"}:
            return 1
        if lowered in {"0", "false", "fail", "failure"}:
            return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def decode_vote_to_label(decoded_vote: str | None) -> int | None:
    if decoded_vote is None:
        return None
    lowered = decoded_vote.strip().lower()
    if lowered in {"1", "success", "pass"}:
        return 1
    if lowered in {"0", "fail", "failure"}:
        return 0
    return None


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype_for_device(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2))


def load_dataset_manifest(path: Path) -> list[DatasetManifestEntry]:
    raw = read_json(path)
    rows = raw["datasets"] if isinstance(raw, dict) else raw
    entries: list[DatasetManifestEntry] = []
    for row in rows:
        root = None if row.get("root") is None else Path(row["root"])
        entries.append(
            DatasetManifestEntry(
                repo_id=row["repo_id"],
                dataset_family=row.get("dataset_family", "custom"),
                split=row.get("split", "train"),
                root=root,
                episode_from=int(row.get("episode_from", 0)),
                episode_to=None if row.get("episode_to") is None else int(row["episode_to"]),
                derived_label=None
                if row.get("derived_label") is None
                else int(row["derived_label"]),
                label_rule=None if row.get("label_rule") is None else str(row["label_rule"]),
                use_for_training=bool(row.get("use_for_training", False)),
                use_for_evaluation=bool(row.get("use_for_evaluation", False)),
            )
        )
    return entries


def select_hook_layer_indices(total_layers: int, num_layers: int = 3) -> list[int]:
    middle_index = total_layers // 2
    remaining_layers = total_layers - middle_index
    step = max(1, remaining_layers // num_layers)
    indices: list[int] = []
    for classifier_index in range(num_layers):
        if classifier_index == num_layers - 1:
            indices.append(total_layers - 1)
        else:
            indices.append(min(middle_index + classifier_index * step, total_layers - 1))
    return indices


class VLMFeatureExtractor:
    def __init__(
        self,
        vlm_model_id: str,
        device: str | None = None,
        hook_layer_indices: list[int] | None = None,
    ) -> None:
        self.vlm_model_id = vlm_model_id
        self.device = torch.device(default_device() if device is None else device)
        self.dtype = torch_dtype_for_device(self.device.type)
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=self.dtype,
        )
        self.model = PeftModel.from_pretrained(base_model, vlm_model_id)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        layers = self._find_language_model_layers()
        if hook_layer_indices is None:
            self.hook_layer_indices = select_hook_layer_indices(len(layers))
        else:
            self.hook_layer_indices = hook_layer_indices

        self.layer_features: dict[int, torch.Tensor] = {}
        self.hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        for feature_index, layer_index in enumerate(self.hook_layer_indices):
            target_layer = layers[layer_index]

            def hook_fn(
                module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor | tuple[torch.Tensor, ...],
                feature_slot: int = feature_index,
            ) -> None:
                if isinstance(output, tuple):
                    self.layer_features[feature_slot] = output[0].detach()
                else:
                    self.layer_features[feature_slot] = output.detach()

            self.hook_handles.append(target_layer.register_forward_hook(hook_fn))

    def _find_language_model_layers(self) -> nn.ModuleList:
        model_root = self.model.base_model.model.model
        if hasattr(model_root, "language_model"):
            language_model = model_root.language_model
            if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
                return language_model.model.layers
            if hasattr(language_model, "layers"):
                return language_model.layers
        if hasattr(model_root, "layers"):
            return model_root.layers
        raise RuntimeError("Could not locate language-model layers in the adapter-backed VLM")

    def extract_episode_features(
        self,
        episode: EpisodeRecord,
        frames_per_camera: int,
    ) -> tuple[list[torch.Tensor], str | None]:
        images: list[Image.Image] = []
        for camera_key in episode.cameras:
            frames = sample_segment_frames(episode.cameras[camera_key], frames_per_camera)
            images.extend(Image.fromarray(frame) for frame in frames)

        prompt = build_prompt(images, episode.task)
        model_inputs = self.processor(
            text=[prompt],
            images=[images],
            return_tensors="pt",
            padding="longest",
        ).to(self.device)

        self.layer_features.clear()
        with torch.no_grad():
            outputs = self.model(**model_inputs)

        if len(self.layer_features) != len(self.hook_layer_indices):
            raise RuntimeError(
                f"Expected {len(self.hook_layer_indices)} hooked feature tensors, "
                f"got {len(self.layer_features)}"
            )

        last_token_logits = outputs.logits[:, -1, :]
        predicted_token_id = torch.argmax(last_token_logits, dim=-1)
        decoded_vote = self.processor.decode(
            [int(predicted_token_id[0].item())], skip_special_tokens=True
        )

        features = [
            self.layer_features[index][0].to(torch.float32).cpu()
            for index in range(len(self.hook_layer_indices))
        ]
        return features, decoded_vote

    def cleanup(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.layer_features.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if self.device.type == "mps":
            torch.mps.empty_cache()


class AttentionPooling(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(features).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(weights * features, dim=1)


class FrozenVLMProbeHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.poolers = nn.ModuleList(
            AttentionPooling(feature_dim=feature_dim, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        )
        self.classifiers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_layers)
        )

    def forward(self, layer_features: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        if len(layer_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} feature tensors, got {len(layer_features)}"
            )

        logits: list[torch.Tensor] = []
        for layer_index, features in enumerate(layer_features):
            pooled = self.poolers[layer_index](features, mask)
            logits.append(self.classifiers[layer_index](pooled).squeeze(-1))
        return torch.stack(logits, dim=1)


class LogitCalibrator(nn.Module):
    def __init__(self, scale: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale * logits + self.bias

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=50, line_search_fn="strong_wolfe")
        criterion = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)


class CachedFeatureDataset(Dataset[tuple[list[torch.Tensor], FeatureIndexEntry]]):
    def __init__(self, feature_dir: Path, entries: list[FeatureIndexEntry]) -> None:
        self.feature_dir = feature_dir
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[list[torch.Tensor], FeatureIndexEntry]:
        entry = self.entries[index]
        payload = torch.load(self.feature_dir / entry.feature_path, map_location="cpu")
        layer_features = [tensor.to(torch.float32) for tensor in payload["layer_features"]]
        return layer_features, entry


class FrozenVLMProbeExecutionScorer:
    def __init__(self, checkpoint: ExecutionCheckpoint, device: str | None = None) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device(default_device() if device is None else device)
        self.extractor = VLMFeatureExtractor(
            vlm_model_id=checkpoint.vlm_model_id,
            device=self.device.type,
            hook_layer_indices=checkpoint.hook_layer_indices,
        )
        self.head = FrozenVLMProbeHead(
            feature_dim=checkpoint.feature_dim,
            num_layers=len(checkpoint.hook_layer_indices),
            hidden_dim=checkpoint.hidden_dim,
            dropout_rate=checkpoint.dropout_rate,
        )
        self.head.load_state_dict(checkpoint.head_state_dict)
        self.head.to(self.device)
        self.head.eval()
        self.calibrator = LogitCalibrator(
            scale=checkpoint.calibration_scale,
            bias=checkpoint.calibration_bias,
        ).to(self.device)
        self.calibrator.eval()

    def score_episode(self, episode: EpisodeRecord) -> ExecutionResult:
        layer_features, decoded_vote = self.extractor.extract_episode_features(
            episode=episode,
            frames_per_camera=self.checkpoint.frames_per_camera,
        )
        tensors = [feature.unsqueeze(0).to(self.device) for feature in layer_features]
        mask = torch.ones((1, layer_features[0].shape[0]), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            layer_logits = self.head(tensors, mask)
            mean_logits = layer_logits.mean(dim=1)
            calibrated_logits = self.calibrator(mean_logits)
            probability = float(torch.sigmoid(calibrated_logits)[0].item())

        return ExecutionResult(
            score=probability,
            probability=probability,
            decoded_vote=decoded_vote,
        )

    def cleanup(self) -> None:
        self.extractor.cleanup()


def save_feature_cache_index(output_dir: Path, index: FeatureCacheIndex) -> None:
    payload = {
        "version": index.version,
        "vlm_model_id": index.vlm_model_id,
        "camera_keys": index.camera_keys,
        "frames_per_camera": index.frames_per_camera,
        "hook_layer_indices": index.hook_layer_indices,
        "feature_dim": index.feature_dim,
        "records": [feature_index_entry_to_json(entry) for entry in index.records],
    }
    write_json(output_dir / "index.json", payload)


def load_feature_cache_index(feature_dir: Path) -> FeatureCacheIndex:
    raw = read_json(feature_dir / "index.json")
    records = [feature_index_entry_from_json(row) for row in raw["records"]]
    return FeatureCacheIndex(
        version=int(raw["version"]),
        vlm_model_id=str(raw["vlm_model_id"]),
        camera_keys=[str(camera_key) for camera_key in raw["camera_keys"]],
        frames_per_camera=int(raw["frames_per_camera"]),
        hook_layer_indices=[int(index) for index in raw["hook_layer_indices"]],
        feature_dim=int(raw["feature_dim"]),
        records=records,
    )


def feature_index_entry_to_json(entry: FeatureIndexEntry) -> dict[str, object]:
    return {
        "feature_path": entry.feature_path,
        "repo_id": entry.repo_id,
        "dataset_family": entry.dataset_family,
        "split": entry.split,
        "use_for_training": entry.use_for_training,
        "use_for_evaluation": entry.use_for_evaluation,
        "episode_index": entry.episode_index,
        "task": entry.task,
        "label": entry.label,
        "quality_score": entry.quality_score,
        "decoded_vote": entry.decoded_vote,
        "num_tokens": entry.num_tokens,
    }


def feature_index_entry_from_json(row: dict[str, object]) -> FeatureIndexEntry:
    label_value = row["label"]
    label = None if label_value is None else int(label_value)
    decoded_vote_value = row["decoded_vote"]
    decoded_vote = None if decoded_vote_value is None else str(decoded_vote_value)
    return FeatureIndexEntry(
        feature_path=str(row["feature_path"]),
        repo_id=str(row["repo_id"]),
        dataset_family=str(row["dataset_family"]),
        split=str(row["split"]),
        use_for_training=bool(row["use_for_training"]),
        use_for_evaluation=bool(row["use_for_evaluation"]),
        episode_index=int(row["episode_index"]),
        task=str(row["task"]),
        label=label,
        quality_score=float(row["quality_score"]),
        decoded_vote=decoded_vote,
        num_tokens=int(row["num_tokens"]),
    )


def save_execution_checkpoint(
    path: Path,
    head: FrozenVLMProbeHead,
    cache_index: FeatureCacheIndex,
    calibrator: LogitCalibrator,
    threshold: float,
) -> None:
    checkpoint = {
        "backend_name": "frozen_vlm_probe",
        "backend_version": 1,
        "vlm_model_id": cache_index.vlm_model_id,
        "camera_keys": cache_index.camera_keys,
        "frames_per_camera": cache_index.frames_per_camera,
        "hook_layer_indices": cache_index.hook_layer_indices,
        "feature_dim": cache_index.feature_dim,
        "hidden_dim": head.hidden_dim,
        "dropout_rate": head.dropout_rate,
        "threshold": threshold,
        "calibration_scale": float(calibrator.scale.detach().cpu().item()),
        "calibration_bias": float(calibrator.bias.detach().cpu().item()),
        "label_mapping": {"fail": 0, "success": 1},
        "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
    }
    torch.save(checkpoint, path)


def load_execution_checkpoint(path: Path) -> ExecutionCheckpoint:
    raw = torch.load(path, map_location="cpu")
    return ExecutionCheckpoint(
        checkpoint_path=path,
        backend_name=str(raw["backend_name"]),
        backend_version=int(raw["backend_version"]),
        vlm_model_id=str(raw["vlm_model_id"]),
        camera_keys=[str(camera_key) for camera_key in raw["camera_keys"]],
        frames_per_camera=int(raw["frames_per_camera"]),
        hook_layer_indices=[int(index) for index in raw["hook_layer_indices"]],
        feature_dim=int(raw["feature_dim"]),
        hidden_dim=int(raw["hidden_dim"]),
        dropout_rate=float(raw["dropout_rate"]),
        threshold=float(raw["threshold"]),
        calibration_scale=float(raw["calibration_scale"]),
        calibration_bias=float(raw["calibration_bias"]),
        label_mapping={str(key): int(value) for key, value in raw["label_mapping"].items()},
        head_state_dict={
            str(key): value.to(torch.float32) for key, value in raw["head_state_dict"].items()
        },
    )


def filter_entries_for_training(
    index: FeatureCacheIndex,
) -> tuple[list[FeatureIndexEntry], list[FeatureIndexEntry]]:
    labeled_entries = [entry for entry in index.records if entry.label is not None]
    train_entries = [
        entry for entry in labeled_entries if entry.use_for_training and entry.split == "train"
    ]
    val_entries = [
        entry for entry in labeled_entries if entry.use_for_training and entry.split == "val"
    ]
    if not train_entries:
        raise ValueError(
            "No labeled training entries with split='train' were found in the feature cache"
        )
    if not val_entries:
        raise ValueError(
            "No labeled validation entries with split='val' were found in the feature cache"
        )
    return train_entries, val_entries


def filter_entries_for_evaluation(index: FeatureCacheIndex) -> list[FeatureIndexEntry]:
    labeled_entries = [
        entry for entry in index.records if entry.label is not None and entry.use_for_evaluation
    ]
    if labeled_entries:
        return labeled_entries
    fallback_entries = [
        entry for entry in index.records if entry.label is not None and entry.split == "test"
    ]
    if fallback_entries:
        return fallback_entries
    raise ValueError("No labeled evaluation entries were found in the feature cache")


def collate_feature_batch(
    batch: list[tuple[list[torch.Tensor], FeatureIndexEntry]],
) -> FeatureBatch:
    if not batch:
        raise ValueError("Feature batch cannot be empty")

    num_layers = len(batch[0][0])
    max_tokens = max(features[0].shape[0] for features, _ in batch)
    feature_dim = batch[0][0][0].shape[1]
    mask = torch.zeros((len(batch), max_tokens), dtype=torch.bool)
    layer_features = [
        torch.zeros((len(batch), max_tokens, feature_dim), dtype=torch.float32)
        for _ in range(num_layers)
    ]
    labels = torch.zeros(len(batch), dtype=torch.float32)
    entries: list[FeatureIndexEntry] = []

    for batch_index, (features, entry) in enumerate(batch):
        token_count = features[0].shape[0]
        mask[batch_index, :token_count] = True
        for layer_index in range(num_layers):
            layer_features[layer_index][batch_index, :token_count] = features[layer_index]
        if entry.label is None:
            raise ValueError("Training and evaluation batches require explicit labels")
        labels[batch_index] = float(entry.label)
        entries.append(entry)

    return FeatureBatch(layer_features=layer_features, mask=mask, labels=labels, entries=entries)


def create_feature_loader(
    feature_dir: Path,
    entries: list[FeatureIndexEntry],
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[list[torch.Tensor], FeatureIndexEntry]]:
    dataset = CachedFeatureDataset(feature_dir=feature_dir, entries=entries)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_feature_batch,
    )


def run_head_on_loader(
    loader: DataLoader[FeatureBatch],
    head: FrozenVLMProbeHead,
    calibrator: LogitCalibrator | None,
    device: torch.device,
) -> tuple[list[float], list[int], list[FeatureIndexEntry], list[float]]:
    probabilities: list[float] = []
    labels: list[int] = []
    entries: list[FeatureIndexEntry] = []
    raw_logits: list[float] = []

    head.eval()
    with torch.no_grad():
        for batch in loader:
            layer_features = [feature.to(device) for feature in batch.layer_features]
            mask = batch.mask.to(device)
            layer_logits = head(layer_features, mask)
            mean_logits = layer_logits.mean(dim=1)
            logits = mean_logits if calibrator is None else calibrator(mean_logits)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            raw_logits.extend(mean_logits.cpu().tolist())
            labels.extend(int(label) for label in batch.labels.tolist())
            entries.extend(batch.entries)

    return probabilities, labels, entries, raw_logits


def fit_execution_head(
    feature_dir: Path,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout_rate: float,
    threshold: float,
    device: str | None = None,
    on_epoch_end: Callable[[dict[str, object]], None] | None = None,
) -> tuple[FeatureCacheIndex, FrozenVLMProbeHead, LogitCalibrator, dict[str, object]]:
    cache_index = load_feature_cache_index(feature_dir)
    train_entries, val_entries = filter_entries_for_training(cache_index)
    train_loader = create_feature_loader(
        feature_dir=feature_dir,
        entries=train_entries,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = create_feature_loader(
        feature_dir=feature_dir,
        entries=val_entries,
        batch_size=batch_size,
        shuffle=False,
    )

    torch_device = torch.device(default_device() if device is None else device)
    head = FrozenVLMProbeHead(
        feature_dim=cache_index.feature_dim,
        num_layers=len(cache_index.hook_layer_indices),
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    ).to(torch_device)

    positive_count = sum(entry.label == 1 for entry in train_entries)
    negative_count = sum(entry.label == 0 for entry in train_entries)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training requires both positive and negative labeled entries")

    pos_weight = torch.tensor(
        negative_count / positive_count,
        dtype=torch.float32,
        device=torch_device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float | None] | None = None
    best_score = float("-inf")
    training_history: list[dict[str, float | None]] = []

    for epoch in range(1, num_epochs + 1):
        head.train()
        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            layer_features = [feature.to(torch_device) for feature in batch.layer_features]
            mask = batch.mask.to(torch_device)
            labels = batch.labels.to(torch_device)

            optimizer.zero_grad()
            layer_logits = head(layer_features, mask)
            expanded_labels = labels.unsqueeze(1).expand_as(layer_logits)
            loss = criterion(layer_logits, expanded_labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch.entries)
            total_examples += len(batch.entries)

        val_probabilities, val_labels, _, val_logits = run_head_on_loader(
            loader=val_loader,
            head=head,
            calibrator=None,
            device=torch_device,
        )
        val_metrics = sanitized_metrics(
            compute_binary_metrics(val_probabilities, val_labels, threshold)
        )
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": total_loss / total_examples,
            **val_metrics,
        }
        training_history.append(epoch_metrics)
        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "epoch": epoch,
                    "train_count": len(train_entries),
                    "val_count": len(val_entries),
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "metrics": epoch_metrics,
                    "history": training_history,
                }
            )

        comparison_score = (
            float(val_metrics["auroc"])
            if val_metrics["auroc"] is not None
            else float(val_metrics["balanced_accuracy"] or 0.0)
        )
        if comparison_score > best_score:
            best_score = comparison_score
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu() for key, value in head.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a checkpointable execution head")

    head.load_state_dict(best_state)
    best_val_probabilities, best_val_labels, _, best_val_logits = run_head_on_loader(
        loader=val_loader,
        head=head,
        calibrator=None,
        device=torch_device,
    )
    calibrator = LogitCalibrator()
    calibration_logits = torch.tensor(best_val_logits, dtype=torch.float32)
    calibration_labels = torch.tensor(best_val_labels, dtype=torch.float32)
    calibrator.fit(calibration_logits, calibration_labels)

    calibrated_probabilities = torch.sigmoid(calibrator(calibration_logits)).tolist()
    calibrated_metrics = sanitized_metrics(
        compute_binary_metrics(calibrated_probabilities, val_labels, threshold)
    )
    metrics = {
        "threshold": threshold,
        "train_count": len(train_entries),
        "val_count": len(val_entries),
        "best_uncalibrated_val_probabilities_mean": sum(best_val_probabilities)
        / len(best_val_probabilities),
        "best_uncalibrated_val_metrics": best_metrics,
        "calibrated_val_metrics": calibrated_metrics,
        "history": training_history,
    }
    return cache_index, head.cpu(), calibrator.cpu(), metrics


def evaluate_execution_checkpoint(
    feature_dir: Path,
    checkpoint: ExecutionCheckpoint,
    batch_size: int,
    device: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cache_index = load_feature_cache_index(feature_dir)
    eval_entries = filter_entries_for_evaluation(cache_index)
    loader = create_feature_loader(
        feature_dir=feature_dir,
        entries=eval_entries,
        batch_size=batch_size,
        shuffle=False,
    )

    torch_device = torch.device(default_device() if device is None else device)
    head = FrozenVLMProbeHead(
        feature_dim=checkpoint.feature_dim,
        num_layers=len(checkpoint.hook_layer_indices),
        hidden_dim=checkpoint.hidden_dim,
        dropout_rate=checkpoint.dropout_rate,
    ).to(torch_device)
    head.load_state_dict(checkpoint.head_state_dict)
    calibrator = LogitCalibrator(
        scale=checkpoint.calibration_scale,
        bias=checkpoint.calibration_bias,
    ).to(torch_device)

    execution_probabilities, labels, entries, _ = run_head_on_loader(
        loader=loader,
        head=head,
        calibrator=calibrator,
        device=torch_device,
    )

    rows: list[dict[str, object]] = []
    quality_probabilities: list[float] = []
    vlm_vote_probabilities: list[float] = []
    for entry, label, execution_probability in zip(
        entries, labels, execution_probabilities, strict=True
    ):
        decoded_vote_label = decode_vote_to_label(entry.decoded_vote)
        decoded_vote_probability = 0.5 if decoded_vote_label is None else float(decoded_vote_label)
        quality_probabilities.append(entry.quality_score)
        vlm_vote_probabilities.append(decoded_vote_probability)
        rows.append(
            {
                "repo_id": entry.repo_id,
                "dataset_family": entry.dataset_family,
                "split": entry.split,
                "episode_index": entry.episode_index,
                "task": entry.task,
                "label": label,
                "execution_probability": execution_probability,
                "predicted_label": int(execution_probability >= checkpoint.threshold),
                "quality_score": entry.quality_score,
                "decoded_vote": entry.decoded_vote,
                "decoded_vote_probability": decoded_vote_probability,
            }
        )

    execution_metrics = sanitized_metrics(
        compute_binary_metrics(execution_probabilities, labels, checkpoint.threshold)
    )
    quality_metrics = sanitized_metrics(
        compute_binary_metrics(quality_probabilities, labels, DEFAULT_THRESHOLD)
    )
    constant_positive_metrics = sanitized_metrics(
        compute_binary_metrics([1.0] * len(labels), labels, DEFAULT_THRESHOLD)
    )
    vlm_vote_metrics = sanitized_metrics(
        compute_binary_metrics(vlm_vote_probabilities, labels, DEFAULT_THRESHOLD)
    )

    family_summaries: dict[str, dict[str, object]] = {}
    families = sorted({entry.dataset_family for entry in entries})
    for family in families:
        family_rows = [row for row in rows if row["dataset_family"] == family]
        family_labels = [int(row["label"]) for row in family_rows]
        family_execution = [float(row["execution_probability"]) for row in family_rows]
        family_quality = [float(row["quality_score"]) for row in family_rows]
        family_summaries[family] = {
            "count": len(family_rows),
            "execution_metrics": sanitized_metrics(
                compute_binary_metrics(family_execution, family_labels, checkpoint.threshold)
            ),
            "quality_metrics": sanitized_metrics(
                compute_binary_metrics(family_quality, family_labels, DEFAULT_THRESHOLD)
            ),
        }

    summary = {
        "backend_name": checkpoint.backend_name,
        "checkpoint_path": str(checkpoint.checkpoint_path),
        "vlm_model_id": checkpoint.vlm_model_id,
        "camera_keys": checkpoint.camera_keys,
        "frames_per_camera": checkpoint.frames_per_camera,
        "threshold": checkpoint.threshold,
        "count": len(rows),
        "execution_metrics": execution_metrics,
        "quality_only_baseline_metrics": quality_metrics,
        "constant_positive_baseline_metrics": constant_positive_metrics,
        "decoded_vote_baseline_metrics": vlm_vote_metrics,
        "dataset_family_summaries": family_summaries,
    }
    return rows, summary


def default_camera_keys_for_backend(
    requested_camera_keys: list[str],
    checkpoint: ExecutionCheckpoint | None,
) -> list[str]:
    if requested_camera_keys:
        return requested_camera_keys
    if checkpoint is not None:
        return checkpoint.camera_keys
    return []
