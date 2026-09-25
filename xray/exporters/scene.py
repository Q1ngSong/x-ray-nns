"""Derive the 3D scene model (lanes, blocks, groups) rendered by the exported player."""

from __future__ import annotations

import base64
import io
from itertools import groupby
from pathlib import Path
import re
from typing import Any, Mapping

from xray.ir import InferenceTrace, TensorRecord

# Numbered siblings such as ``encoder.layers.3`` fold into one collapsible group.
_NUMBERED = re.compile(r"^(?P<parent>.+)\.\d+$")
_KIND_ORDER = {"input": 0, "module": 1, "result": 2}
# Tensor semantic types that feed a panel of the block owning their producer.
_VIEW_KINDS = {"attention": "attention", "mlp_activation": "mlp", "query": "q", "key": "k", "value": "v",
               "cross_attention": "cross"}
# Semantic types whose raw payload a view reads; recorders with ``raw_scope="views"`` keep only these.
VIEW_SEMANTICS = frozenset(_VIEW_KINDS)
# Most token vectors a PCA basis is fitted on; larger sets (large feature maps) are strided down to it.
_PCA_FIT = 65536


def build_scene(trace: InferenceTrace, bundle: Path) -> dict[str, Any]:
    """3D scene model: lanes, blocks and groups derived from a trace's module hierarchy. [主线]

    Blocks are the selected hierarchy boundaries, ``input.*`` checkpoints and
    results; lanes follow ``metadata.branch``.  Inside a lane, inputs come first
    and every other block follows its first runtime event, so the layout never
    replaces the recorded order with a topological sort.  Lanes stack in the
    order another lane first consumes their output, then by their first event.

    Args:
        trace: Recorded trace; operations may carry ``branch``, ``stage``,
            ``selected``, ``owner``, ``label`` and ``group`` metadata supplied by the model adapter.
        bundle: Bundle directory; ``inputs.image.asset_path`` is resolved inside it.
    Returns:
        dict[str, Any] — JSON-compatible ``lanes`` (ordered ``{"group"|"block": id}``
        items), ``blocks``, ``groups``, ``op_blocks`` (operation → owning block),
        ``links`` (``[from, to]`` blocks in different lanes joined by a recorded Tensor),
        ``image`` (centre-crop PNG data URL, or ``None``) and ``views`` (block →
        intermediate-value payloads, see ``_views``).

    变更: 2026-09-24 块可由 ``label`` 命名、由 ``group`` 归入指定组（原先只按编号兄弟分组）；新增 ``links``，
        跨通道连线改按真实 Tensor 流向给出，原先由页面把各通道末尾连到结果通道。
    变更: 2026-09-24 通道先按输出被别的通道首次使用的先后排，原先只按首个事件；CLIP 仍是 vision、text、fusion。
    """
    tensors = {tensor.id: tensor for tensor in trace.tensors}
    operations = {operation.id: operation for operation in trace.operations}
    first_seen: dict[str, int] = {}
    for index, event in enumerate(trace.events):
        for key in (event.operation, event.tensor):
            if key is not None:
                first_seen.setdefault(key, index)
    never = len(trace.events)
    blocks: dict[str, dict[str, Any]] = {}
    rank: dict[str, tuple[int, int]] = {}
    for operation in trace.operations:
        meta = operation.metadata or {}
        lane = str(meta.get("branch") or "model")
        stage = str(meta.get("stage") or "")
        if meta.get("selected") and operation.module:
            block_id, kind, label = operation.module, "module", str(meta.get("label") or operation.module.rsplit(".", 1)[-1])
        elif operation.level == "checkpoint" and stage.startswith("input."):
            block_id, kind, label = f"{lane}:{stage}", "input", stage.removeprefix("input.")
        elif operation.level == "result":
            block_id, kind, label = f"{lane}:{stage or operation.name}", "result", operation.name
        else:
            continue
        # Checkpoints of model inputs only consume their Tensor, so fall back to inputs.
        tensor_id = next(iter(operation.outputs or operation.inputs), None)
        touched = min(first_seen.get(key, never) for key in (operation.id, tensor_id, meta.get("derived_from")) if key)
        if block_id not in blocks:
            tensor = tensors.get(tensor_id) if tensor_id else None
            blocks[block_id] = {
                "id": block_id, "lane": lane, "kind": kind, "label": label, "stage": stage,
                "operations": [], "tensor": tensor_id, "shape": list(tensor.shape) if tensor else [],
                "semantic": (tensor.semantic_type if tensor else None) or meta.get("semantic_type"),
            }
            if meta.get("group"):
                blocks[block_id]["parent"] = str(meta["group"])
            rank[block_id] = (_KIND_ORDER[kind], touched)
        blocks[block_id]["operations"].append(operation.id)
        rank[block_id] = min(rank[block_id], (_KIND_ORDER[kind], touched))

    op_blocks = {op_id: block["id"] for block in blocks.values() for op_id in block["operations"]}
    for operation in trace.operations:
        if operation.id in op_blocks:
            continue
        owner = (operation.metadata or {}).get("owner")
        path = owner if owner in blocks else operation.module or ""
        # Leaf modules belong to the nearest selected ancestor, e.g. ``layers.3.mlp.fc1``.
        while path and path not in blocks:
            path = path.rpartition(".")[0]
        if path:
            op_blocks[operation.id] = path
    links = _links(trace, tensors, operations, blocks, op_blocks)
    # Lanes stack in the order their outputs are first consumed by another lane, so a lane sits next to
    # the one it feeds; the rest follow by their first event.
    feeds: dict[str, int] = {}
    for index, (source, _target) in enumerate(links):
        feeds.setdefault(blocks[source]["lane"], index)

    by_lane: dict[str, list[str]] = {}
    for block_id in sorted(blocks, key=rank.__getitem__):
        by_lane.setdefault(blocks[block_id]["lane"], []).append(block_id)
    groups: dict[str, dict[str, Any]] = {}
    lanes = []
    for lane in sorted(by_lane, key=lambda name: (feeds.get(name, len(links)), min(rank[block_id][1] for block_id in by_lane[name]))):
        items: list[dict[str, str]] = []
        keyed = [(_group_parent(blocks[block_id]), block_id) for block_id in by_lane[lane]]
        for parent, members in groupby(keyed, key=lambda item: item[0]):
            member_ids = [block_id for _, block_id in members]
            if parent and len(member_ids) > 1 and parent not in groups:
                groups[parent] = {"id": parent, "lane": lane, "label": ".".join(parent.split(".")[-2:]),
                                  "blocks": member_ids}
                for block_id in member_ids:
                    blocks[block_id]["group"] = parent
                items.append({"group": parent})
            else:
                items.extend({"block": block_id} for block_id in member_ids)
        lanes.append({"id": lane, "items": items})

    scene = {"lanes": lanes, "blocks": blocks, "groups": groups, "op_blocks": op_blocks, "links": links,
             "image": _input_image(trace.inputs.get("image"), bundle)}
    scene["views"] = _views(trace, bundle, scene)
    return scene


def _links(trace: InferenceTrace, tensors: Mapping[str, TensorRecord], operations: Mapping[str, Any],
           blocks: Mapping[str, Mapping[str, Any]], op_blocks: Mapping[str, str]) -> list[list[str]]:
    """Cross-lane links: blocks of different lanes joined by a Tensor one produced and the other consumed. [主线]

    A result block that re-publishes a Tensor produced elsewhere (such as CLIP's ``image_embeds``)
    stands for that Tensor, so its lane links from the result rather than from the producing module.
    Each source block links once per target lane, to the first consumer in recorded order.

    Args:
        trace: Recorded trace; operations are scanned in recorded order.
        tensors: Tensor ID → record, for producers.
        operations: Operation ID → record, for the results' inputs and outputs.
        blocks: Scene blocks by ID, for their lanes.
        op_blocks: Operation → owning block from ``build_scene``.
    Returns:
        list[list[str]] — ``[source block, target block]`` pairs.
    """
    represented: dict[str, str] = {}
    for block in blocks.values():
        for op_id in block["operations"] if block["kind"] == "result" else ():
            operation = operations[op_id]
            if not operation.outputs and operation.inputs:
                represented.setdefault(operation.inputs[0], block["id"])
    links, linked = [], set()
    for operation in trace.operations:
        target = op_blocks.get(operation.id)
        if target is None:
            continue
        lane = blocks[target]["lane"]
        for tensor_id in operation.inputs:
            producer = tensors[tensor_id].producer if tensor_id in tensors else None
            source = represented.get(tensor_id) or op_blocks.get(producer or "")
            if source and blocks[source]["lane"] != lane and (source, lane) not in linked:
                linked.add((source, lane))
                links.append([source, target])
    return links


def _views(trace: InferenceTrace, bundle: Path, scene: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Intermediate-value views: per-block panel payloads derived from raw ``tensors/*.pt``. [主线]

    Layer outputs and Q/K/V become one RGB colour per token, using a PCA basis
    shared by every block of the same lane and kind so colours stay comparable
    across layers; a ``[batch, channels, H, W]`` feature map counts its H × W
    positions as tokens and keeps its ``grid``.  Attention keeps every head;
    cross-attention keeps one spatial map per head and token; MLP activations
    become the share of positive units per token; final outputs keep their values.

    Args:
        trace: Recorded trace whose binary ``storage.path`` entries point into ``bundle``.
        bundle: Bundle directory; missing payloads or paths outside it are skipped.
        scene: Structure from ``build_scene``; its ``blocks`` and ``op_blocks`` locate each source.
    Returns:
        dict[str, dict[str, Any]] — block id → ``hidden`` / ``attention`` / ``cross`` / ``mlp`` /
        ``qkv`` (``q``, ``k``, ``v``) / ``output`` payloads; empty without torch.

    变更: 2026-09-24 4 维模块输出按空间位置出 ``hidden``（带 ``grid``、不带 ``norm``），新增 ``cross``；
        结果运算的 ``threshold`` / ``detected`` / ``score_type`` 随 ``output`` 带出。原有 3 维输出与各面板不变。
    """
    try:
        import torch
    except ImportError:  # Raw payloads only exist for torch recordings.
        return {}
    blocks, root = scene["blocks"], bundle.resolve()
    operations = {operation.id: operation for operation in trace.operations}
    sources = []
    for block in blocks.values():
        if block["kind"] == "module" and len(block["shape"]) in (3, 4):
            sources.append((block["id"], "hidden", block["tensor"]))
        if block["kind"] == "result" or block["semantic"] == "embedding_projection":
            sources.append((block["id"], "output", block["tensor"]))
    for tensor in trace.tensors:
        kind, block_id = _VIEW_KINDS.get(tensor.semantic_type or ""), scene["op_blocks"].get(tensor.producer or "")
        if kind and block_id:
            sources.append((block_id, kind, tensor.id))

    records = {tensor.id: tensor for tensor in trace.tensors}
    views: dict[str, dict[str, Any]] = {}
    token_states: dict[tuple[str, str, int], list[tuple[str, str, Any, list[int] | None]]] = {}
    for block_id, kind, tensor_id in sources:
        value = _load_raw(records.get(tensor_id), root)
        if value is None:
            continue
        if kind in ("hidden", "q", "k", "v"):
            grid = [int(side) for side in value.shape[-2:]] if value.dim() == 4 else None
            if grid:
                value = value.flatten(2).transpose(1, 2)
            states = value.reshape(-1, value.shape[-2], value.shape[-1]) if value.dim() >= 2 else value.reshape(1, 1, -1)
            token_states.setdefault((blocks[block_id]["lane"], kind, int(states.shape[-1])), []).append(
                (block_id, tensor_id, states, grid))
        elif kind == "attention":
            views.setdefault(block_id, {})["attention"] = _attention_view(tensor_id, value)
        elif kind == "cross":
            views.setdefault(block_id, {})["cross"] = _cross_view(tensor_id, value, records[tensor_id].metadata)
        elif kind == "mlp":
            views.setdefault(block_id, {})["mlp"] = _mlp_view(tensor_id, value)
        else:
            meta = operations[blocks[block_id]["operations"][0]].metadata or {}
            view = _output_view(tensor_id, value, blocks[block_id]["semantic"] == "similarity_logits")
            view.update({key: meta[key] for key in ("threshold", "detected", "score_type") if key in meta})
            views.setdefault(block_id, {})["output"] = view
    for (_lane, kind, _width), members in token_states.items():
        colours = _pca_rgb([states for _, _, states, _ in members])
        for (block_id, tensor_id, states, grid), rgb in zip(members, colours):
            payload = {"tensor": tensor_id, "rows": int(states.shape[0]), "cols": int(states.shape[1]),
                       "rgb": base64.b64encode(rgb).decode("ascii")}
            if grid:
                payload["grid"] = grid
            else:
                payload["norm"] = [round(float(norm), 3) for norm in torch.linalg.vector_norm(states, dim=-1).reshape(-1)]
            if kind == "hidden":
                views.setdefault(block_id, {})["hidden"] = payload
            else:
                views.setdefault(block_id, {}).setdefault("qkv", {})[kind] = payload
    return views


def _load_raw(record: TensorRecord | None, root: Path) -> Any | None:
    """Raw payload: load a recorded ``tensors/*.pt`` file as a float32 CPU Tensor. [主线]

    Args:
        record: Tensor record; only ``storage.kind == "binary"`` payloads are read.
        root: Resolved bundle directory; paths escaping it are ignored.
    Returns:
        Any | None — float32 Tensor, or ``None`` when absent or not floating point (e.g. token ids).
    """
    import torch

    storage = record.storage if record else None
    if storage is None or storage.kind != "binary" or not storage.path:
        return None
    path = (root / storage.path).resolve()
    if root not in path.parents or not path.is_file():
        return None
    value = torch.load(path, map_location="cpu", weights_only=True)
    return value.float() if torch.is_tensor(value) and value.is_floating_point() else None


def _pca_rgb(states: list[Any]) -> list[bytes]:
    """PCA colours: map every token vector to RGB along three components shared by all ``states``. [主线]

    Tokens are centred and L2-normalised first, so the residual stream's growing
    norm does not decide the colours of late layers; each component's sign is
    fixed by its largest loading, so reruns give the same colours.  Beyond
    ``_PCA_FIT`` tokens (large feature maps) the basis is fitted on an even
    stride of them and applied to all.

    Args:
        states: float Tensors ``[rows, tokens, features]`` with one feature width.
    Returns:
        list[bytes] — per state, ``rows * tokens * 3`` uint8 RGB values in row-major order.

    变更: 2026-09-24 超过 ``_PCA_FIT`` 个 token 时按等距抽样拟合基，归一化改为原地运算；此前全部 token 参与，
        不超过该数的输入（CLIP 各运行）颜色不变。
    """
    import torch

    centred = torch.cat([state.reshape(-1, state.shape[-1]) for state in states])
    centred.sub_(centred.mean(dim=1, keepdim=True))
    centred.div_(torch.linalg.vector_norm(centred, dim=1, keepdim=True).clamp_min(1e-6))
    centred.sub_(centred.mean(dim=0))
    basis = torch.linalg.svd(centred[::(len(centred) - 1) // _PCA_FIT + 1], full_matrices=False).Vh[:3]
    signs = torch.sign(basis.gather(1, basis.abs().argmax(dim=1, keepdim=True)))
    coords = centred @ (basis * torch.where(signs == 0, torch.ones_like(signs), signs)).T
    coords = torch.nn.functional.pad(coords, (0, 3 - coords.shape[1]))
    low, high = torch.quantile(coords, 0.02, dim=0), torch.quantile(coords, 0.98, dim=0)
    rgb = ((coords - low) / (high - low).clamp_min(1e-6)).clamp(0, 1).mul(255).round().to(torch.uint8)
    sizes = [state.shape[0] * state.shape[1] for state in states]
    return [bytes(chunk.reshape(-1).tolist()) for chunk in rgb.split(sizes)]


def _attention_view(tensor_id: str, attention: Any) -> dict[str, Any]:
    """Attention view: every head as a uint8 map scaled by that map's own maximum. [主线]

    Args:
        tensor_id: Source Tensor ID, opened in the inspector when the panel is clicked.
        attention: float Tensor ``[batch, heads, queries, tokens]``; ``queries`` is the token count
            unless only leading query rows (such as CLS) were kept. An older head-averaged
            ``[batch, tokens, tokens]`` recording counts as one head.
    Returns:
        dict[str, Any] — ``rows`` (batch), ``heads``, ``size`` (tokens), ``queries``, base64 ``data`` in
        batch-head-query-key order, and ``scale`` (the weight that 255 stands for, per map).

    变更: 2026-09-24 新增 ``queries``：只存 CLS 查询行的注意力也能画；完整方阵照旧，``queries`` 等于 ``size``。
    """
    import torch

    maps = attention if attention.dim() == 4 else attention.reshape(-1, 1, attention.shape[-2], attention.shape[-1])
    scale = maps.amax(dim=(-2, -1)).clamp_min(1e-12)
    data = (maps / scale[..., None, None]).clamp(0, 1).mul(255).round().to(torch.uint8)
    return {"tensor": tensor_id, "rows": int(maps.shape[0]), "heads": int(maps.shape[1]), "size": int(maps.shape[-1]),
            "queries": int(maps.shape[-2]), "data": base64.b64encode(bytes(data.reshape(-1).tolist())).decode("ascii"),
            "scale": [round(float(value), 6) for value in scale.reshape(-1)]}


def _cross_view(tensor_id: str, probs: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-attention view: one spatial map per head and prompt token, each scaled by its own maximum. [主线]

    Args:
        tensor_id: Source Tensor ID, opened in the inspector when the panel is clicked.
        probs: float Tensor ``[1, heads, H * W, tokens]`` of attention probabilities over prompt tokens.
        metadata: Tensor metadata with ``grid`` (``[H, W]``) and ``tokens`` (one string per column).
    Returns:
        dict[str, Any] — ``heads``, ``grid``, ``tokens`` (labels), base64 ``data`` in head-token-position
        order and ``scale`` (the probability that 255 stands for, per map).
    """
    import torch

    maps = probs[0].transpose(-1, -2)
    scale = maps.amax(dim=-1).clamp_min(1e-12)
    data = (maps / scale[..., None]).clamp(0, 1).mul(255).round().to(torch.uint8)
    labels = [str(token) for token in metadata.get("tokens", [])][:maps.shape[1]]
    return {"tensor": tensor_id, "heads": int(maps.shape[0]), "grid": [int(side) for side in metadata["grid"]],
            "tokens": labels + [""] * (maps.shape[1] - len(labels)),
            "data": base64.b64encode(bytes(data.reshape(-1).tolist())).decode("ascii"),
            "scale": [round(float(value), 6) for value in scale.reshape(-1)]}


def _mlp_view(tensor_id: str, activation: Any) -> dict[str, Any]:
    """MLP view: per token, the share of hidden units whose activation is positive. [主线]

    Args:
        tensor_id: Source Tensor ID of the activation-function output.
        activation: float Tensor ``[batch, tokens, units]`` after the MLP nonlinearity.
    Returns:
        dict[str, Any] — ``rows``, ``cols``, base64 uint8 ``active`` (share × 255) and per-token ``mean``.
    """
    import torch

    states = activation.reshape(-1, activation.shape[-2], activation.shape[-1])
    active = (states > 0).float().mean(dim=-1)
    return {"tensor": tensor_id, "rows": int(states.shape[0]), "cols": int(states.shape[1]),
            "active": base64.b64encode(bytes(active.mul(255).round().to(torch.uint8).reshape(-1).tolist())).decode("ascii"),
            "mean": [round(float(value), 4) for value in states.mean(dim=-1).reshape(-1)]}


def _output_view(tensor_id: str, value: Any, logits: bool) -> dict[str, Any]:
    """Output view: final vectors (embeddings, logits) as rounded rows, plus softmax for logits. [主线]

    Args:
        tensor_id: Source Tensor ID of the result.
        value: float Tensor; leading dimension is the batch (image or prompt), the rest is flattened.
        logits: True for similarity logits, which also get a row-wise softmax.
    Returns:
        dict[str, Any] — ``values`` (rows of floats) and, for logits, ``probs``.
    """
    rows = value.reshape(value.shape[0], -1) if value.dim() > 1 else value.reshape(1, -1)
    view: dict[str, Any] = {"tensor": tensor_id, "values": [[round(float(item), 4) for item in row] for row in rows]}
    if logits:
        view["probs"] = [[round(float(item), 4) for item in row] for row in rows.softmax(dim=-1)]
    return view


def _group_parent(block: Mapping[str, Any]) -> str | None:
    """Group key: the block's declared ``parent``, else the parent path of a numbered module block. [主线]

    Args:
        block: Scene block; only ``kind == "module"`` blocks can be grouped by number.
    Returns:
        str | None — parent module path, or ``None`` when the block stays ungrouped.

    变更: 2026-09-24 先看块自带的 ``parent``（适配器用 ``group`` 指定），没有时仍按编号兄弟分组。
    """
    if block.get("parent"):
        return block["parent"]
    match = _NUMBERED.match(block["id"]) if block["kind"] == "module" else None
    return match.group("parent") if match else None


def _input_image(image: Any, bundle: Path) -> str | None:
    """Input image: centre-cropped 224 px PNG data URL usable as a WebGL texture. [主线]

    Browsers treat ``file://`` images as cross-origin and WebGL refuses them as
    textures, while an embedded data URL is same-origin.  The square crop matches
    CLIP's resize-short-side then centre-crop preprocessing shown by the 2D view.

    Args:
        image: ``trace.inputs["image"]`` — mapping whose ``asset_path`` is relative
            to the bundle; paths escaping the bundle are ignored.
        bundle: Bundle directory that holds the copied input asset.
    Returns:
        str | None — ``data:image/png;base64,...``, or ``None`` without an image or Pillow.
    """
    asset = image.get("asset_path") if isinstance(image, Mapping) else None
    if not isinstance(asset, str):
        return None
    root = bundle.resolve()
    path = (root / asset).resolve()
    if root not in path.parents or not path.is_file():
        return None
    try:
        from PIL import Image
    except ImportError:  # Pillow ships with the ``clip`` extra; the 3D view falls back to a plain plane.
        return None
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    side = min(rgb.size)
    left, top = (rgb.width - side) // 2, (rgb.height - side) // 2
    crop = rgb.crop((left, top, left + side, top + side)).resize((224, 224))
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
