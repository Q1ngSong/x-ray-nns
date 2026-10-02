"""Record torchvision image classifiers, including executed residual additions."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import operator
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn

from xray.exporters import export_bundle
from xray.recorder.torch import TorchModuleRecorder


class _FunctionCall(nn.Module):
    """Expose an FX function call to the existing module recorder. [基础设施]"""

    def __init__(self, function: Any) -> None:
        super().__init__()
        self.function = function

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the original function with its unchanged arguments. [主线]

        Args:
            args: Original positional Tensor/scalar arguments from the FX graph.
            kwargs: Original keyword arguments from the FX graph.
        Returns:
            Original function result, without detaching or recomputing it.
        """
        return self.function(*args, **kwargs)


def prepare_cnn(model: nn.Module) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Expose CNN calls, residual add and flatten as individually recorded nodes. [主线]

    FX keeps the evaluation computation; function nodes become hooked modules and
    repeated ReLU calls get distinct module instances. Unsupported function nodes
    fail explicitly. This is a capture plan for AlexNet and ResNet, not arbitrary FX.

    Args:
        model: Torchvision AlexNet or ResNet in evaluation mode, without attached hooks.
    Returns:
        Executable GraphModule and per-call scene annotations.

    变更: 2026-10-01 AlexNet 按五个 Conv 划分阶段，记录真实 kernel/stride/padding；计算不变。
    """
    graph = torch.fx.symbolic_trace(model)
    annotations: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    feature_stage = 0
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"):
            continue
        original = str(node.target)
        stack = list(node.meta.get("nn_module_stack", {}))
        scope = next((path for path in reversed(stack) if re.fullmatch(r"layer[1-4]\.\d+", path)), None)
        if node.op == "call_function":
            if node.target not in (operator.add, torch.flatten):
                raise ValueError(f"unsupported CNN function: {node.target}")
            label = "Residual add" if node.target is operator.add else "Flatten"
            module = _FunctionCall(node.target)
            original = (scope + "." if scope else "") + node.name
            target = "fx_" + node.name
            graph.add_module(target, module)
            node.op, node.target = "call_module", target
        elif node.op == "call_module":
            module = graph.get_submodule(original)
            label = module.__class__.__name__
            if original in seen:
                target = "fx_" + node.name
                graph.add_module(target, deepcopy(module))
                node.target = target
            seen.add(original)
            scope = re.match(r"(layer[1-4]\.\d+)(?:\.|$)", original)
            scope = scope.group(1) if scope else None
        else:
            raise ValueError(f"unsupported CNN node: {node.op}")
        meta: dict[str, Any] = {"selected": True, "branch": "model", "stage": original,
                                "label": label, "description": f"{original} · {module}",
                                "semantic_type": "feature_map" if isinstance(module, nn.Conv2d) else None}
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
            meta["spatial"] = {}
            for key in ("kernel_size", "stride", "padding"):
                value = getattr(module, key)
                meta["spatial"][key] = list(value) if isinstance(value, tuple) else [value, value]
        if scope:
            meta["group"] = scope
            meta["label"] = "Add (+)" if label == "Residual add" else original.removeprefix(scope + ".")
        elif original.startswith("features."):
            if isinstance(module, nn.Conv2d):
                feature_stage += 1
            meta["group"] = "features.stage" + str(feature_stage)
            meta["label"] = f"{original.split('.')[-1]} {label}"
        elif original.startswith("classifier."):
            meta["group"] = "classifier"
            meta["label"] = f"{original.split('.')[-1]} {label}"
        if label == "Residual add":
            meta["semantic_type"] = "residual_add"
            meta["description"] = "Residual addition: the transformed branch and shortcut are added elementwise before ReLU. Both input tensors are recorded."
        annotations[str(node.target)] = meta
    graph.graph.lint()
    graph.recompile()
    return graph, annotations


def run_cnn(architecture: str, model_path: str | Path, image_path: str | Path,
            output_dir: str | Path) -> Path:
    """Record a pretrained AlexNet/ResNet-18 CPU forward and export offline playback. [主线]

    变更: 2026-10-01 AlexNet 引导文字对应五个独立卷积组及池化尺寸。

    Args:
        architecture: `alexnet` or `resnet18`, using torchvision ImageNet-1K V1 preprocessing.
        model_path: Local official V1 state dict; the published SHA-256 prefix is verified.
        image_path: Local RGB image; the transformed centre crop is saved for spatial alignment.
        output_dir: New/empty output directory relative to the caller's working directory.
    Returns:
        Bundle path containing the real trace, raw tensors, scene and offline HTML.
    """
    import torchvision
    from torchvision import models
    from PIL import Image
    from xray.exporters.scene import build_scene
    import json

    builders = {"alexnet": (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),
                "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1)}
    builder, weights = builders[architecture]
    checkpoint, destination = Path(model_path), Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose a new or empty CNN output directory.")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = weights.url.rsplit("-", 1)[-1].split(".")[0]
    if not digest.startswith(expected):
        raise ValueError(f"Expected official {architecture} V1 weights with SHA-256 prefix {expected}")
    model = builder(weights=None).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    image = Image.open(image_path).convert("RGB")
    transform = weights.transforms()
    pixels = transform(image).unsqueeze(0)
    graph, annotations = prepare_cnn(model)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with torch.no_grad():
        reference = model(pixels.clone())
    destination.mkdir(parents=True, exist_ok=True)
    assets = destination / "assets"
    assets.mkdir(exist_ok=True)
    # Undo only normalization, keeping the exact resize and centre crop the model received.
    mean = torch.tensor(transform.mean).view(3, 1, 1)
    std = torch.tensor(transform.std).view(3, 1, 1)
    torchvision.transforms.functional.to_pil_image((pixels[0] * std + mean).clamp(0, 1)).save(assets / "input-image.png")
    with TorchModuleRecorder(graph, destination, model_name=architecture,
                             trace_id=architecture + "-" + destination.name, raw_limit_bytes=16_000_000,
                             module_annotations=annotations, selected_modules=annotations) as recorder:
        image_id = recorder.record_checkpoint("image", pixels, semantic_type="image_pixels", branch="model",
                                              stage="input.image", metadata={"description": "224 × 224 centre crop; model input is normalized with ImageNet mean and standard deviation."})
        with torch.no_grad():
            logits = graph(pixels)
        torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-6)
        error = float((logits - reference).abs().max())
        recorder.record_result("ImageNet classification", logits, semantic_type="classification_logits", branch="model",
                               stage="classification", metadata={"labels": weights.meta["categories"],
                               "description": "ImageNet-1K class logits. The top-five display uses softmax over all 1000 classes; omitted classes retain their probability mass."})
        trace = recorder.build(inputs={"image": {"asset_path": "assets/input-image.png", "size": [224, 224], "caption": "ImageNet preprocessing: resize to 256, centre crop to 224 × 224, then normalize RGB."},
                                      "image_pixels": {"tensor_id": image_id},
                                      "summary": [["Architecture", architecture], ["Weights", "ImageNet-1K V1"], ["Input", "224 × 224 centre crop"]],
                                      "guide": (["展开 Conv 1：11×11 卷积用 stride 4 将 224×224 变成 55×55，MaxPool 再缩到 27×27。",
                                                 "Conv 2 的池化得到 13×13；Conv 3、4 保持空间大小，Conv 5 后池化为 6×6。"] if architecture == "alexnet" else
                                                ["先展开 Stage 1，再展开 layer1.0：棕色跳连把输入送到 Add (+)，与主分支相加。",
                                                 "先展开 Stage 2，再展开 layer2.0：downsample 用投影匹配通道与空间尺寸，再做残差相加。"])
                                                + ["Layer output 的颜色是空间特征的 PCA 摘要；Feature channels 显示单个通道。",
                                                   "展开 Classifier 并点击 Scores，查看 ImageNet 前五类预测，概率按全部 1000 类计算。"]},
                               metadata={"executed_model": True, "provenance": "real_forward",
                                         "backend": "torchvision." + architecture, "weights": {"url": weights.url, "sha256": digest},
                                         "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__},
                                         "capture": {"dataflow_edges": True, "feature_channels": True},
                                         "validation": {"reference_max_abs_error": error},
                                         "preprocessing": str(transform)})
    trace.metadata["capture"].update({"granularity": "fx_call_modules_and_functions", "functional_operations": True,
                                      "unhooked_functional_operations": False,
                                      "coverage_note": "Executed torchvision eval graph; residual additions and flatten are captured as module boundaries. Parameters and convolution internals are not separate nodes."})
    export_bundle(trace, destination)
    scene = build_scene(trace, destination)
    (destination / "scene.json").write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"model": architecture, "operations": len(trace.operations), "tensors": len(trace.tensors),
                      "events": len(trace.events), "reference_max_abs_error": error,
                      "top1": weights.meta["categories"][int(logits.argmax())]}, ensure_ascii=False))
    return destination
