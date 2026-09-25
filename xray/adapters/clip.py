"""CLIP ViT configuration and semantic hints for the first recorder path."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClipConfig:
    """Capture the architecture facts needed to label a CLIP trace. [主线]"""

    model_path: Path
    projection_dim: int
    vision_hidden_size: int
    vision_layers: int
    vision_heads: int
    image_size: int
    patch_size: int
    text_hidden_size: int
    text_layers: int
    text_heads: int
    context_length: int


class ClipAdapter:
    """Read a local Hugging Face CLIP config and classify common Tensor roles. [主线]"""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"CLIP config.json not found under {path}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if data.get("model_type") != "clip":
            raise ValueError(f"expected a CLIP config, got model_type={data.get('model_type')!r}")
        vision = data["vision_config"]
        text = data["text_config"]
        self.config = ClipConfig(
            model_path=path,
            projection_dim=int(data["projection_dim"]),
            vision_hidden_size=int(vision["hidden_size"]),
            vision_layers=int(vision["num_hidden_layers"]),
            vision_heads=int(vision["num_attention_heads"]),
            image_size=int(vision["image_size"]),
            patch_size=int(vision["patch_size"]),
            text_hidden_size=int(text["hidden_size"]),
            text_layers=int(text["num_hidden_layers"]),
            text_heads=int(text["num_attention_heads"]),
            context_length=int(text["max_position_embeddings"]),
        )

    def semantic_type(self, module_name: str, shape: tuple[int, ...] = ()) -> str | None:
        """Map a module path and optional shape to a renderer hint. [主线]

        Args:
            module_name: str — fully qualified CLIP module path from the recorder.
            shape: tuple[int, ...] — observed Tensor shape, used only for attention disambiguation.
        Returns:
            str | None — semantic renderer key, or `None` when generic rendering is safer.

        变更: 2026-09-23 `mlp.activation_fn` 的输出标为 `mlp_activation`，原先为 `None`。
        """
        name = module_name.lower()
        if "patch_embedding" in name:
            return "patch_embedding"
        if "position_embedding" in name:
            return "position_embedding"
        if "token_embedding" in name or name.endswith("text_model.embeddings"):
            return "token_embedding"
        if "self_attn" in name:
            for projection, role in (("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value")):
                if name.endswith(projection):
                    return role
            if name.endswith("self_attn") and len(shape) == 4:
                return "attention"  # Known attention module output; shape alone is insufficient.
            return "attention_output"
        if "visual_projection" in name or "text_projection" in name:
            return "embedding_projection"
        if "layer_norm" in name or "layernorm" in name:
            return "normalization"
        if "logit_scale" in name:
            return "similarity_scale"
        if name.endswith("mlp.activation_fn"):
            return "mlp_activation"
        return None

    def describe(self) -> dict[str, Any]:
        """Return a serializable architecture summary for trace metadata. [基础设施]

        Returns:
            dict[str, Any] — local model path and CLIP ViT dimensions.
        """
        config = self.config
        return {
            "model_type": "clip",
            "model_path": str(config.model_path),
            "projection_dim": config.projection_dim,
            "vision": {
                "hidden_size": config.vision_hidden_size,
                "layers": config.vision_layers,
                "heads": config.vision_heads,
                "image_size": config.image_size,
                "patch_size": config.patch_size,
            },
            "text": {
                "hidden_size": config.text_hidden_size,
                "layers": config.text_layers,
                "heads": config.text_heads,
                "context_length": config.context_length,
            },
        }
