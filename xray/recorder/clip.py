"""Run and record one local Hugging Face CLIP inference."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from xray.adapters import ClipAdapter
from xray.exporters import export_bundle
from xray.recorder.torch import TorchModuleRecorder


def run_clip(
    model_path: str | Path,
    image_path: str | Path,
    texts: Sequence[str],
    output_dir: str | Path,
    *,
    device: str = "auto",
    save_raw: bool = True,
    trace_id: str = "clip-run",
) -> Path:
    """Execute local CLIP once and export its module trace as HTML. [主线]

    Args:
        model_path: str | Path — local Hugging Face CLIP directory.
        image_path: str | Path — local RGB image used as the image input.
        texts: Sequence[str] — one or more text prompts paired with the image.
        output_dir: str | Path — bundle directory receiving HTML, metadata, and tensors.
        device: str — `auto`, `cpu`, or `mps` on Apple Silicon.
        save_raw: bool — whether to save bounded intermediate tensors as `.pt` files.
        trace_id: str — stable identifier written to the manifest.
    Returns:
        Path — generated bundle directory containing `index.html`.

    变更: 2026-09-21 记录本地真实 forward 的 runtime、配置哈希和叶子模块采集边界。
    变更: 2026-09-21 增加选定层级 hooks、真实 token/patch checkpoints 和 fusion 输入连接。
    变更: 2026-09-21 改用 eager attention 并开启 output_attentions，新增逐层 head-averaged 注意力 checkpoint。
    变更: 2026-09-23 注意力 checkpoint 改存逐 head 张量 [batch, heads, tokens, tokens] 并记录所属层 owner；
        2D preview 仍是第一个样本的 head 平均。
    变更: 2026-09-23 超过 CLIP 上下文长度（77 token）的 prompt 按 tokenizer 截断，原先会在 forward 时报错。
    变更: 2026-09-24 ``asset_path`` 写成正斜杠，Windows 上录的 bundle 在其他系统里也能找到输入图。
    """
    try:
        import torch
        from PIL import Image
        import transformers
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise RuntimeError(
            "CLIP recording needs the optional runtime; run `python -m pip install -e '.[clip]'` in xraynns."
        ) from error

    model_root = Path(model_path)
    image_file = Path(image_path)
    destination = Path(output_dir)
    if not texts:
        raise ValueError("at least one --text value is required")
    if not image_file.is_file():
        raise FileNotFoundError(f"image not found: {image_file}")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this machine")

    image = Image.open(image_file).convert("RGB")
    config_hash = hashlib.sha256((model_root / "config.json").read_bytes()).hexdigest()
    processor = CLIPProcessor.from_pretrained(model_root, local_files_only=True)
    model = CLIPModel.from_pretrained(
        model_root,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device).eval()
    inputs = processor(text=list(texts), images=image, return_tensors="pt", padding=True, truncation=True)
    model_inputs = {name: value.to(device) for name, value in inputs.items()}
    selected_modules, module_annotations = _clip_capture_plan(model)
    destination.mkdir(parents=True, exist_ok=True)
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    image_target = assets / f"input-image{image_file.suffix.lower() or '.png'}"
    shutil.copy2(image_file, image_target)
    adapter = ClipAdapter(model_root)
    with TorchModuleRecorder(
        model,
        destination,
        model_name="CLIPModel",
        semantic_adapter=adapter,
        trace_id=trace_id,
        save_raw=save_raw,
        selected_modules=selected_modules,
        module_annotations=module_annotations,
    ) as recorder:
        with torch.no_grad():
            # Request the per-layer attention maps explicitly. Module hooks
            # capture layer inputs/outputs, but attention probabilities are
            # returned by the CLIP model only through this opt-in output.
            outputs = model(**model_inputs, output_attentions=True)

        module_names = {name for name, _module in model.named_modules()}
        for branch, model_output in (
            ("vision", getattr(outputs, "vision_model_output", None)),
            ("text", getattr(outputs, "text_model_output", None)),
        ):
            attentions = getattr(model_output, "attentions", None)
            for layer_index, attention in enumerate(attentions or ()):
                if attention is None:
                    continue
                # The raw payload keeps every head for the 3D view; the inline
                # preview stays the head-averaged map of the first batch item.
                heads = torch.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
                matrix = heads[0].mean(dim=0).detach().to("cpu", dtype=torch.float32)
                sequence = int(matrix.shape[-1])
                metadata = {
                    "layer_index": layer_index,
                    "head_count": int(attention.shape[1]),
                    "sequence_length": sequence,
                    "preview_reduction": "mean_heads",
                }
                owner = f"{branch}_model.encoder.layers.{layer_index}"
                if owner in module_names:
                    metadata["owner"] = owner
                recorder.record_checkpoint(
                    f"{branch}.attention.layer.{layer_index}",
                    heads,
                    semantic_type="attention",
                    branch=branch,
                    stage=f"{branch}.attention.layer.{layer_index}",
                    metadata=metadata,
                    preview_override={
                        "kind": "attention_map",
                        "rows": sequence,
                        "cols": sequence,
                        "values": [
                            _finite_float(value)
                            for value in matrix.reshape(-1).tolist()
                        ],
                    },
                )

        # Token checkpoints are derived from the tensors produced by the real
        # forward.  They are represented as bounded raw Tensor payloads plus a
        # semantic preview so the browser can show patch grids and token text.
        text_token_tensor_id = None
        patch_token_tensor_id = None
        pixel_tensor_id = None
        text_input = recorder.latest_module_tensor("text_model.embeddings", inputs=True)
        if text_input is not None:
            _text_input_tensor_id, text_ids = text_input
            token_ids = text_ids.detach().to("cpu").tolist()
            token_strings = [
                processor.tokenizer.convert_ids_to_tokens(row)
                for row in token_ids
            ]
            text_token_tensor_id = recorder.record_checkpoint(
                "inputs.text_tokens",
                text_ids,
                semantic_type="text_tokens",
                branch="text",
                stage="input.tokens",
                metadata={
                    "token_ids": token_ids,
                    "token_strings": token_strings,
                    "sequence_count": len(token_ids),
                },
                preview_override={
                    "kind": "token_ids",
                    "values": token_ids,
                    "tokens": token_strings,
                },
            )

        vision_embedding = recorder.latest_module_tensor("vision_model.embeddings")
        if vision_embedding is not None:
            embedding_tensor_id, embedding_value = vision_embedding
            if embedding_value.ndim == 3 and embedding_value.shape[1] > 1:
                patch_tokens = embedding_value[:, 1:, :]
                patch_norms = patch_tokens.detach().to("cpu", dtype=torch.float32).norm(dim=-1)
                patch_norm_values = [
                    _finite_float(value)
                    for value in patch_norms.reshape(-1).tolist()[:49]
                ]
                patch_token_tensor_id = recorder.record_checkpoint(
                    "inputs.image_patch_tokens",
                    patch_tokens,
                    semantic_type="image_patch_tokens",
                    branch="vision",
                    stage="input.patch_tokens",
                    source_tensor_id=embedding_tensor_id,
                    metadata={
                        "source_tensor_id": embedding_tensor_id,
                        "special_tokens": 1,
                        "grid_shape": [7, 7],
                        "token_count": int(patch_tokens.shape[1]),
                    },
                    preview_override={
                        "kind": "patch_grid_norms",
                        "rows": 7,
                        "cols": 7,
                        "values": patch_norm_values,
                    },
                )

        pixel_input = recorder.latest_module_tensor("vision_model.embeddings", inputs=True)
        if pixel_input is not None:
            _pixel_tensor_id, pixel_values = pixel_input
            pixel_tensor_id = recorder.record_checkpoint(
                "inputs.image_pixels",
                pixel_values,
                semantic_type="image_pixels",
                branch="vision",
                stage="input.image",
                metadata={"asset_path": image_target.relative_to(destination).as_posix()},
            )
        result_tensor_ids: dict[str, str] = {}
        for result_name, branch in (("image_embeds", "vision"), ("text_embeds", "text")):
            result = getattr(outputs, result_name, None)
            if result is not None:
                tensor_id = recorder.record_result(
                    result_name, result, semantic_type="embedding", branch=branch, stage="projection.output"
                )
                if tensor_id is not None:
                    result_tensor_ids[result_name] = tensor_id
        for result_name in ("logits_per_image", "logits_per_text"):
            result = getattr(outputs, result_name, None)
            if result is not None:
                recorder.record_result(
                    result_name,
                    result,
                    semantic_type="similarity_logits",
                    branch="fusion",
                    stage="fusion.similarity",
                    input_tensor_ids=(
                        result_tensor_ids.get("image_embeds"),
                        result_tensor_ids.get("text_embeds"),
                    ),
                )
        input_descriptors: dict[str, Any] = {
            "image": {"asset_path": image_target.relative_to(destination).as_posix(), "size": list(image.size), "mode": image.mode},
            "text": list(texts),
        }
        if pixel_tensor_id is not None:
            input_descriptors["image_pixels"] = {"tensor_id": pixel_tensor_id, "shape": list(model_inputs["pixel_values"].shape)}
        if patch_token_tensor_id is not None:
            input_descriptors["image_patch_tokens"] = {"tensor_id": patch_token_tensor_id, "grid_shape": [7, 7], "token_count": 49}
        if text_token_tensor_id is not None:
            input_descriptors["text_tokens"] = {"tensor_id": text_token_tensor_id, "sequence_count": len(texts)}
        trace = recorder.build(
            inputs=input_descriptors,
            metadata={
                "provenance": "real_forward",
                "executed_model": True,
                "backend": "transformers.CLIPModel",
                "model_path": str(model_root),
                "device": device,
                "model": adapter.describe(),
                "model_config_sha256": config_hash,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                "capture": {
                    "granularity": "leaf_and_selected_module_hooks",
                    "functional_operations": False,
                    "parameter_tensors": False,
                    "branches": ["vision", "text", "fusion"],
                    "selected_stages": [
                        "input.image", "input.patch_tokens", "vision.embeddings",
                        "vision.encoder", "vision.projection", "input.tokens",
                        "text.embeddings", "text.encoder", "text.projection",
                        "fusion.similarity", "vision.attention", "text.attention",
                    ],
                    "attention": {
                        "captured": True,
                        "heads": "all",
                        "preview_reduction": "mean_heads",
                        "layers": {"vision": adapter.config.vision_layers, "text": adapter.config.text_layers},
                    },
                },
            },
        )
    return export_bundle(trace, destination)


def _clip_capture_plan(model: Any) -> tuple[tuple[str, ...], dict[str, Mapping[str, Any]]]:
    """Build selected CLIP module paths and semantic branch annotations. [主线]

    Args:
        model: Any — loaded ``transformers.CLIPModel`` instance.
    Returns:
        tuple[tuple[str, ...], dict[str, Mapping[str, Any]]] — exact module
        paths present in this model and bounded annotations for each operation.
    """
    names = {name for name, _module in model.named_modules()}
    annotations: dict[str, Mapping[str, Any]] = {}
    for name in sorted(names):
        branch: str | None = None
        stage: str | None = None
        semantic_type: str | None = None
        if name == "vision_model.embeddings":
            branch, stage, semantic_type = "vision", "vision.embeddings", "image_patch_embeddings"
        elif name == "vision_model.pre_layrnorm":
            branch, stage, semantic_type = "vision", "vision.pre_norm", "vision_tokens"
        elif name.startswith("vision_model.encoder.layers."):
            layer = name.removeprefix("vision_model.encoder.layers.")
            if layer.isdigit():
                branch, stage, semantic_type = "vision", f"vision.encoder.layer.{int(layer):02d}", "vision_hidden_states"
        elif name == "vision_model.post_layernorm":
            branch, stage, semantic_type = "vision", "vision.post_norm", "vision_pooled"
        elif name == "visual_projection":
            branch, stage, semantic_type = "vision", "vision.projection", "image_embedding"
        elif name == "text_model.embeddings":
            branch, stage, semantic_type = "text", "text.embeddings", "text_embeddings"
        elif name.startswith("text_model.encoder.layers."):
            layer = name.removeprefix("text_model.encoder.layers.")
            if layer.isdigit():
                branch, stage, semantic_type = "text", f"text.encoder.layer.{int(layer):02d}", "text_hidden_states"
        elif name == "text_model.final_layer_norm":
            branch, stage, semantic_type = "text", "text.post_norm", "text_pooled"
        elif name == "text_projection":
            branch, stage, semantic_type = "text", "text.projection", "text_embedding"
        if branch is not None:
            annotations[name] = {
                "branch": branch,
                "stage": stage,
                "semantic_type": semantic_type,
                "selected": True,
            }
    return tuple(annotations), annotations


def _finite_float(value: Any) -> float | None:
    """Convert a numeric preview value to a finite JSON scalar. [基础设施]

    Args:
        value: Any — candidate numeric value from a bounded Tensor preview.
    Returns:
        float | None — finite scalar, or ``None`` for NaN and infinity.
    """
    import math

    number = float(value)
    return number if math.isfinite(number) else None
