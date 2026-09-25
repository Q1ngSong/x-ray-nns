"""PyTorch module-hook recorder for one local inference."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from xray.exporters.scene import VIEW_SEMANTICS
from xray.ir import (
    InferenceTrace,
    OperationRecord,
    TensorRecord,
    TensorStats,
    TensorStorage,
)
from xray.recorder.runtime import TraceRecorder


@dataclass
class _TensorState:
    """Mutable capture state assembled into a TensorRecord at the end of a run. [基础设施]"""

    id: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    producer: str | None
    consumers: list[str]
    semantic_type: str | None
    stats: TensorStats | None
    storage: TensorStorage | None
    metadata: dict[str, Any]


class TorchModuleRecorder:
    """Capture leaf-module inputs, outputs, statistics, and runtime events. [主线]"""

    def __init__(
        self,
        model: Any,
        output_dir: str | Path,
        *,
        model_name: str | None = None,
        semantic_adapter: Any | None = None,
        trace_id: str = "torch-run",
        save_raw: bool = True,
        raw_limit_bytes: int = 4_000_000,
        raw_scope: str = "all",
        preview_limit: int = 256,
        selected_modules: Iterable[str] | None = None,
        module_annotations: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Initialize bounded capture state for one model. [主线]

        Args:
            model: Any — PyTorch module whose leaf calls should be observed.
            output_dir: str | Path — bundle directory for bounded Tensor payloads.
            model_name: str | None — display name; defaults to the model class.
            semantic_adapter: Any | None — optional module-to-semantic mapper.
            trace_id: str — stable trace identifier.
            save_raw: bool — persist bounded CPU Tensor payloads when true.
            raw_limit_bytes: int — maximum raw payload size per Tensor.
            raw_scope: str — ``all`` saves every bounded Tensor; ``views`` only those a
                scene view reads: selected-module boundaries, checkpoints, results and
                ``VIEW_SEMANTICS`` types. Other values raise ValueError.
            preview_limit: int — maximum preview values per Tensor.
            selected_modules: Iterable[str] | None — non-leaf module paths to hook
                in addition to every leaf module.
            module_annotations: Mapping[str, Mapping[str, Any]] | None — bounded
                operation metadata keyed by module path (for example branch/stage).

        变更: 2026-09-24 新增 ``raw_scope`` 与可切换的 ``paused``（暂停时模块调用不记录）；默认值下行为不变。
        """
        if raw_limit_bytes < 0:
            raise ValueError("raw_limit_bytes must be non-negative")
        if preview_limit < 0:
            raise ValueError("preview_limit must be non-negative")
        if raw_scope not in ("all", "views"):
            raise ValueError("raw_scope must be 'all' or 'views'")
        self.model = model
        self.output_dir = Path(output_dir)
        self.tensors_dir = self.output_dir / "tensors"
        self.model_name = model_name or model.__class__.__name__
        self.semantic_adapter = semantic_adapter
        self.save_raw = save_raw
        self.raw_limit_bytes = raw_limit_bytes
        self.raw_scope = raw_scope
        # While paused the hooks stay registered but module calls go unrecorded;
        # callers flip it between calls, never inside a hooked module.
        self.paused = False
        self.preview_limit = preview_limit
        self.selected_modules = frozenset(selected_modules or ())
        self.module_annotations = {
            str(name): dict(annotation)
            for name, annotation in (module_annotations or {}).items()
        }
        self._recorder = TraceRecorder(self.model_name, trace_id=trace_id)
        self._module_names = {id(module): name for name, module in model.named_modules()}
        self._hooks: list[Any] = []
        self._pending: dict[int, list[str]] = defaultdict(list)
        self._operations: dict[str, dict[str, Any]] = {}
        self._operation_order: list[str] = []
        self._tensor_states: dict[str, _TensorState] = {}
        self._tensor_ids: dict[int, str] = {}
        self._tensor_refs: dict[int, Any] = {}
        self._tensor_values: dict[str, Any] = {}
        self._tensor_counter = 0
        self._operation_counter = 0
        self._result_tensors: dict[str, str] = {}
        self._leaf_module_count = 0
        self._selected_module_count = 0
        self._assembled = False

    def __enter__(self) -> "TorchModuleRecorder":
        """Register hooks for leaf modules and prepare bundle directories. [主线]

        Returns:
            TorchModuleRecorder — this active recorder.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensors_dir.mkdir(parents=True, exist_ok=True)
        for name, module in self.model.named_modules():
            is_leaf = not any(module.children())
            is_selected = name in self.selected_modules
            if name and (is_leaf or is_selected):
                # Keyword arguments carry important masks and positional metadata in
                # transformer modules, so retain them at the module boundary.
                self._hooks.append(module.register_forward_pre_hook(self._before_module, with_kwargs=True))
                self._hooks.append(module.register_forward_hook(self._after_module))
                if is_leaf:
                    self._leaf_module_count += 1
                if is_selected and not is_leaf:
                    self._selected_module_count += 1
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Remove all hooks after the model call. [基础设施]"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _before_module(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Record module inputs before its forward call. [主线]

        Args:
            module: Any — hooked PyTorch module.
            args: tuple[Any, ...] — positional forward arguments.
            kwargs: Mapping[str, Any] | None — keyword forward arguments.

        变更: 2026-09-21 retain selected non-leaf module annotations alongside leaf hooks.
        变更: 2026-09-24 ``paused`` 时直接返回，不建运算；配对的输出 hook 因无待处理运算而跳过。
        """
        if self.paused:
            return
        operation_id = f"op_{self._operation_counter:05d}"
        self._operation_counter += 1
        module_name = self._module_names.get(id(module), module.__class__.__name__)
        values: list[Any] = list(args)
        if kwargs:
            values.extend(kwargs.values())
        input_ids = tuple(
            self._observe_tensor(value, consumer=operation_id, module_name=module_name)
            for value in _flatten_tensors(values)
        )
        self._operations[operation_id] = {
            "type": module.__class__.__name__,
            "name": module_name or module.__class__.__name__,
            "module": module_name,
            "inputs": list(dict.fromkeys(input_ids)),
            "outputs": [],
            "level": "module",
            "metadata": self._operation_metadata(module_name),
        }
        self._operation_order.append(operation_id)
        for tensor_id in dict.fromkeys(input_ids):
            self._recorder.emit("consume_tensor", tensor=tensor_id, operation=operation_id)
        self._pending[id(module)].append(operation_id)

    def _after_module(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        """Record module outputs after its forward call. [主线]

        Args:
            module: Any — hooked PyTorch module.
            args: tuple[Any, ...] — positional arguments supplied to forward.
            output: Any — possibly nested forward result.
        """
        pending = self._pending[id(module)]
        if not pending:
            return
        operation_id = pending.pop()
        state = self._operations[operation_id]
        output_ids = tuple(
            self._observe_tensor(value, producer=operation_id, module_name=state["module"])
            for value in _flatten_tensors(output)
        )
        state["outputs"] = list(dict.fromkeys(output_ids))
        self._recorder.emit("execute_operation", operation=operation_id)
        for tensor_id in dict.fromkeys(output_ids):
            self._recorder.emit("produce_tensor", tensor=tensor_id, operation=operation_id)

    def record_result(
        self,
        name: str,
        value: Any,
        *,
        semantic_type: str | None = None,
        branch: str | None = None,
        stage: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        input_tensor_ids: Iterable[str] | None = None,
    ) -> str | None:
        """Register a final model output that is outside leaf-module hooks. [主线]

        Args:
            name: str — public result name such as `logits_per_image`.
            value: Any — scalar or Tensor returned by the model.
            semantic_type: str | None — renderer hint for this result.
            branch: str | None — logical result branch, such as ``fusion``.
            stage: str | None — semantic stage label for playback.
            metadata: Mapping[str, Any] | None — bounded result annotations.
            input_tensor_ids: Iterable[str] | None — existing branch outputs
                consumed by a functional result such as CLIP similarity logits.
        Returns:
            str | None — Tensor ID when `value` contains a Tensor.

        变更: 2026-09-21 connect fusion results to their image/text embedding inputs.
        """
        tensors = tuple(_flatten_tensors((value,)))
        if not tensors:
            return None
        tensor = tensors[0]
        tensor_id = self._observe_tensor(tensor, module_name="model_output", semantic_type=semantic_type)
        if semantic_type is not None:
            # The public ModelOutput name is the most useful semantic label for
            # the browser, even when the same Tensor was already seen at a
            # projection module boundary.
            self._tensor_states[tensor_id].semantic_type = semantic_type
            self._tensor_states[tensor_id].metadata["result_name"] = name
        self._result_tensors[name] = tensor_id
        operation_metadata = dict(metadata or {})
        operation_metadata.update({
            key: value for key, value in (
                ("branch", branch),
                ("stage", stage),
                ("semantic_type", semantic_type),
                ("kind", "result"),
            ) if value is not None
        })
        should_record_result = (
            self._tensor_states[tensor_id].producer is None
            or branch is not None
            or stage is not None
            or bool(metadata)
        )
        if should_record_result:
            operation_id = f"op_result_{self._operation_counter:05d}"
            self._operation_counter += 1
            self._operation_order.append(operation_id)
            existing_producer = self._tensor_states[tensor_id].producer is not None
            linked_inputs = [
                input_id for input_id in (input_tensor_ids or ())
                if input_id in self._tensor_states and input_id != tensor_id
            ]
            operation_inputs = ([tensor_id] if existing_producer else []) + linked_inputs
            operation_outputs = [] if existing_producer else [tensor_id]
            self._operations[operation_id] = {
                "type": "ModelOutput",
                "name": name,
                "module": self.model_name,
                "inputs": operation_inputs,
                "outputs": operation_outputs,
                "level": "result",
                "metadata": operation_metadata,
            }
            if existing_producer:
                if operation_id not in self._tensor_states[tensor_id].consumers:
                    self._tensor_states[tensor_id].consumers.append(operation_id)
            for input_id in dict.fromkeys(linked_inputs):
                if operation_id not in self._tensor_states[input_id].consumers:
                    self._tensor_states[input_id].consumers.append(operation_id)
            for input_id in dict.fromkeys(operation_inputs):
                self._recorder.emit("consume_tensor", tensor=input_id, operation=operation_id)
            if not existing_producer:
                self._tensor_states[tensor_id].producer = operation_id
            self._recorder.emit("execute_operation", operation=operation_id)
            if not existing_producer:
                self._recorder.emit("produce_tensor", tensor=tensor_id, operation=operation_id)
        return tensor_id

    def record_checkpoint(
        self,
        name: str,
        value: Any,
        *,
        semantic_type: str | None = None,
        branch: str | None = None,
        stage: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        preview_override: Mapping[str, Any] | None = None,
        source_tensor_id: str | None = None,
    ) -> str | None:
        """Add an inspectable checkpoint without overwriting Tensor producers. [主线]

        A checkpoint is a lightweight graph operation.  When ``value`` is a
        Tensor already observed at a module boundary, the operation consumes
        that existing edge and leaves its producer untouched.  A newly derived
        view can be represented as a checkpoint output; callers may provide a
        ``source_tensor_id`` to make that derivation explicit.

        Args:
            name: str — stable display name for the checkpoint.
            value: Any — Tensor (or nested value whose first Tensor is used).
            semantic_type: str | None — renderer hint such as ``patch_tokens``.
            branch: str | None — logical branch label, for example ``vision``.
            stage: str | None — ordered semantic stage label.
            metadata: Mapping[str, Any] | None — bounded JSON-compatible details.
            preview_override: Mapping[str, Any] | None — semantic preview payload.
            source_tensor_id: str | None — observed Tensor that produced a new view.
        Returns:
            str | None — captured Tensor ID, or ``None`` for scalar-only values.
        """
        tensors = tuple(_flatten_tensors((value,)))
        if not tensors:
            return None
        tensor = tensors[0]
        tensor_id = self._observe_tensor(
            tensor,
            module_name="checkpoint",
            semantic_type=semantic_type,
            metadata_override=metadata,
            preview_override=preview_override,
        )
        state = self._tensor_states[tensor_id]
        operation_id = f"op_checkpoint_{self._operation_counter:05d}"
        self._operation_counter += 1
        operation_metadata = dict(metadata or {})
        operation_metadata.update({
            key: value for key, value in (
                ("branch", branch),
                ("stage", stage),
                ("semantic_type", semantic_type),
                ("kind", "checkpoint"),
            ) if value is not None
        })
        if source_tensor_id is not None:
            operation_metadata["derived_from"] = source_tensor_id
        # Existing tensors already have an authoritative producer.  Consume
        # them for inspection instead of assigning a second, invalid producer.
        existing_producer = state.producer is not None
        operation_inputs: list[str] = []
        operation_outputs: list[str] = []
        if source_tensor_id is not None and source_tensor_id in self._tensor_states:
            operation_inputs.append(source_tensor_id)
            source_state = self._tensor_states[source_tensor_id]
            if operation_id not in source_state.consumers:
                source_state.consumers.append(operation_id)
        # A Tensor that was consumed before the checkpoint is an external model
        # input.  It has no producer in this trace, and the inspection operation
        # must not pretend that the checkpoint created it.
        is_external_input = state.producer is None and bool(state.consumers)
        if (
            not existing_producer
            and not is_external_input
            and (source_tensor_id is None or source_tensor_id != tensor_id)
        ):
            operation_outputs.append(tensor_id)
            state.producer = operation_id
        else:
            operation_inputs.append(tensor_id)
            if operation_id not in state.consumers:
                state.consumers.append(operation_id)
        self._operations[operation_id] = {
            "type": "Checkpoint",
            "name": name,
            "module": "checkpoint",
            "inputs": list(dict.fromkeys(operation_inputs)),
            "outputs": operation_outputs,
            "level": "checkpoint",
            "metadata": operation_metadata,
        }
        self._operation_order.append(operation_id)
        for input_id in dict.fromkeys(operation_inputs):
            self._recorder.emit("consume_tensor", tensor=input_id, operation=operation_id)
        self._recorder.emit("execute_operation", operation=operation_id)
        for output_id in operation_outputs:
            self._recorder.emit("produce_tensor", tensor=output_id, operation=operation_id)
        return tensor_id

    def latest_module_tensor(self, module_name: str, *, inputs: bool = False) -> tuple[str, Any] | None:
        """Return the latest captured Tensor of a module path: its first output, or its first input. [基础设施]

        The lookup follows the operation graph rather than Python object IDs,
        which is useful when a framework wrapper creates an equivalent input
        object while forwarding into a selected module.

        Args:
            module_name: str — exact ``named_modules`` path.
            inputs: bool — read the first input of the module call instead of its first output.
        Returns:
            tuple[str, Any] | None — Tensor ID and live Tensor, if captured.

        变更: 2026-09-23 并入原 ``latest_module_input_tensor``（改为 ``inputs=True``）；从未用到的
            ``output_index`` / ``input_index`` 参数去掉，始终取第一个。
        """
        key = "inputs" if inputs else "outputs"
        for operation_id in reversed(self._operation_order):
            operation = self._operations[operation_id]
            if operation.get("module") != module_name:
                continue
            tensor_ids = operation.get(key, [])
            if not tensor_ids:
                return None
            value = self._tensor_values.get(tensor_ids[0])
            if value is not None:
                return tensor_ids[0], value
        return None

    def build(
        self,
        *,
        inputs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> InferenceTrace:
        """Finalize a validated InferenceTrace from the captured runtime state. [主线]

        Args:
            inputs: dict[str, Any] — serializable descriptions of model inputs.
            metadata: dict[str, Any] — run provenance, device, and result references.
        Returns:
            InferenceTrace — graph and event stream ready for HTML export.

        变更: 2026-09-21 include capture granularity, branch coverage, and selected stage metadata.
        """
        if not self._assembled:
            for operation_id in self._operation_order:
                state = self._operations[operation_id]
                self._recorder.add_operation(
                    OperationRecord(
                        operation_id,
                        state["type"],
                        state["name"],
                        module=state["module"],
                        inputs=tuple(state["inputs"]),
                        outputs=tuple(state["outputs"]),
                        level=state["level"],
                        metadata=state.get("metadata", {}),
                    )
                )
            for state in self._tensor_states.values():
                self._recorder.add_tensor(
                    TensorRecord(
                        state.id,
                        state.shape,
                        state.dtype,
                        device=state.device,
                        producer=state.producer,
                        consumers=tuple(state.consumers),
                        semantic_type=state.semantic_type,
                        stats=state.stats,
                        storage=state.storage,
                        metadata=state.metadata,
                    )
                )
            self._assembled = True
        enriched_metadata = dict(metadata)
        enriched_metadata["results"] = {
            name: {"tensor_id": tensor_id} for name, tensor_id in self._result_tensors.items()
        }
        capture_metadata = dict(enriched_metadata.get("capture") or {})
        capture_metadata.update({
            "operations": len(self._operations),
            "tensors": len(self._tensor_states),
            "leaf_modules": self._leaf_module_count,
            "selected_modules": self._selected_module_count,
            "granularity": (
                "leaf_and_selected_module_hooks"
                if self.selected_modules
                else "leaf_module_hooks"
            ),
            "unhooked_functional_operations": True,
            "coverage_note": "Functional tensor operations between module hooks are not represented as operations.",
        })
        enriched_metadata["capture"] = capture_metadata
        for state in self._tensor_states.values():
            if state.producer is None:
                # A missing producer may be a model input or an unhooked functional
                # operation; do not label it as an external input without provenance.
                state.metadata.setdefault("origin", "unattributed")
        return self._recorder.build(inputs=inputs, metadata=enriched_metadata)

    def _observe_tensor(
        self,
        tensor: Any,
        *,
        producer: str | None = None,
        consumer: str | None = None,
        module_name: str = "",
        semantic_type: str | None = None,
        metadata_override: Mapping[str, Any] | None = None,
        preview_override: Mapping[str, Any] | None = None,
    ) -> str:
        """Assign an ID and persist metadata for one runtime Tensor. [基础设施]

        Args:
            tensor: Any — Tensor leaf observed at a module boundary.
            producer: str | None — operation producing this Tensor version.
            consumer: str | None — operation consuming this Tensor version.
            module_name: str — module path used for semantic hints.
            semantic_type: str | None — explicit renderer hint.
            metadata_override: Mapping[str, Any] | None — bounded metadata merged
                into the Tensor record.
            preview_override: Mapping[str, Any] | None — semantic preview replacing
                the generic flattened-value preview.
        Returns:
            str — stable Tensor ID for the current version.

        变更: 2026-09-21 preserve derived checkpoint metadata without replacing tensor producers.
        """
        key = id(tensor)
        tensor_id = self._tensor_ids.get(key)
        previous_id = tensor_id
        # A module can return an input tensor in-place or as an alias.  The IR has
        # one producer per Tensor, therefore capture a new version at each later
        # producer boundary instead of leaving an output with a stale producer.
        if (
            tensor_id is None
            or producer is not None
            and (
                self._tensor_states[tensor_id].producer is not None
                or bool(self._tensor_states[tensor_id].consumers)
            )
            and self._tensor_states[tensor_id].producer != producer
        ):
            tensor_id = f"tensor_{self._tensor_counter:05d}"
            self._tensor_counter += 1
            self._tensor_ids[key] = tensor_id
            self._tensor_refs[key] = tensor
            self._tensor_states[tensor_id] = self._capture_tensor(tensor, tensor_id, module_name, semantic_type)
            self._tensor_values[tensor_id] = tensor
            if previous_id is not None:
                self._tensor_states[tensor_id].metadata["alias_of"] = previous_id
        state = self._tensor_states[tensor_id]
        if producer is not None and state.producer is None:
            state.producer = producer
        if consumer is not None and consumer not in state.consumers:
            state.consumers.append(consumer)
        if semantic_type is not None and state.semantic_type is None:
            state.semantic_type = semantic_type
        if metadata_override:
            state.metadata.update(dict(metadata_override))
        if preview_override:
            state.metadata["preview"] = dict(preview_override)
        return tensor_id

    def _operation_metadata(self, module_name: str) -> dict[str, Any]:
        """Return a bounded annotation for one module operation. [基础设施]

        Args:
            module_name: str — named module path receiving the annotation.
        Returns:
            dict[str, Any] — branch, stage, and semantic hints for playback.
        """
        annotation = dict(self.module_annotations.get(module_name, {}))
        if self.semantic_adapter is not None and "semantic_type" not in annotation:
            semantic_type = self.semantic_adapter.semantic_type(module_name, ())
            if semantic_type:
                annotation["semantic_type"] = semantic_type
        return annotation

    def _capture_tensor(self, tensor: Any, tensor_id: str, module_name: str, semantic_type: str | None) -> _TensorState:
        """Compute bounded statistics, preview values, and optional raw storage. [基础设施]

        Args:
            tensor: Any — detached or live PyTorch Tensor to snapshot.
            tensor_id: str — destination Tensor ID used for raw storage naming.
            module_name: str — module path used for adapter hints.
            semantic_type: str | None — explicit semantic renderer hint.
        Returns:
            _TensorState — immutable-shape metadata with bounded payload references.

        变更: 2026-09-21 统计改在 clamp 到 ±1e4 的副本上计算，避免 attention mask 的
            -3.4e38 哨兵值令 min/max/mean 溢出；preview 与 raw 仍取原始值。
        变更: 2026-09-24 ``raw_scope="views"`` 时只为视图会读的 Tensor 存 raw；``storage.path``
            一律写成正斜杠，原先 Windows 上录的 bundle 在其他系统里找不到 ``tensors\\*.pt``。
        """
        import torch

        cpu_tensor = tensor.detach().to("cpu")
        if getattr(cpu_tensor, "is_sparse", False):
            cpu_tensor = cpu_tensor.to_dense()
        shape = tuple(int(value) for value in cpu_tensor.shape)
        dtype = str(cpu_tensor.dtype).replace("torch.", "")
        adapter_hint = None
        if self.semantic_adapter is not None and module_name:
            adapter_hint = self.semantic_adapter.semantic_type(module_name, shape)
        flat = cpu_tensor.reshape(-1)
        # Statistics use magnitudes for complex values; all other dtypes are
        # converted to float32 so integer, bool, and half tensors share a path.
        numeric = flat.abs().to(torch.float32) if flat.dtype.is_complex else flat.to(torch.float32)
        raw_numeric = numeric
        nan_count = int(torch.isnan(raw_numeric).sum().item()) if raw_numeric.dtype.is_floating_point else 0
        inf_count = int(torch.isinf(raw_numeric).sum().item()) if raw_numeric.dtype.is_floating_point else 0
        # Attention masks may contain the finite sentinel ``-3.4e38``;
        # clamping the statistics path prevents reductions from overflowing
        # while preserving the original Tensor in optional raw storage.
        numeric = torch.nan_to_num(raw_numeric, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
        sample = numeric[: max(self.preview_limit, 100_000)]
        finite_mask = torch.isfinite(sample)
        finite = sample[finite_mask]
        if finite.numel():
            stats = TensorStats(
                minimum=float(finite.min().item()),
                maximum=float(finite.max().item()),
                mean=float(finite.mean().item()),
                median=float(finite.median().item()),
                std=float(finite.std(unbiased=False).item()),
                norm=float(torch.linalg.vector_norm(finite).item()),
                sparsity=float((finite == 0).float().mean().item()),
                nan_count=nan_count,
                inf_count=inf_count,
            )
        else:
            stats = TensorStats(nan_count=nan_count, inf_count=inf_count)
        preview_values = [
            (float(value) if math.isfinite(float(value)) else None)
            for value in raw_numeric[: self.preview_limit].tolist()
        ]
        metadata: dict[str, Any] = {
            "numel": int(cpu_tensor.numel()),
            "memory_bytes": int(cpu_tensor.numel() * cpu_tensor.element_size()),
            "preview": {"shape": list(shape), "values": preview_values},
            "stats_sampled_numel": int(sample.numel()),
            "stats_truncated": int(sample.numel()) < int(cpu_tensor.numel()),
        }
        storage = TensorStorage(kind="preview")
        viewed = (self.raw_scope == "all" or module_name in self.selected_modules
                  or module_name in ("checkpoint", "model_output") or (semantic_type or adapter_hint) in VIEW_SEMANTICS)
        if self.save_raw and viewed and metadata["memory_bytes"] <= self.raw_limit_bytes:
            relative_path = f"tensors/{tensor_id}.pt"
            # Cloning avoids serializing an entire backing storage when a captured
            # module output is a small view into a much larger tensor.
            torch.save(cpu_tensor.contiguous().clone(), self.output_dir / relative_path)
            storage = TensorStorage(kind="binary", path=relative_path, full=True)
        return _TensorState(
            id=tensor_id,
            shape=shape,
            dtype=dtype,
            device=str(tensor.device),
            producer=None,
            consumers=[],
            semantic_type=semantic_type or adapter_hint,
            stats=stats,
            storage=storage,
            metadata=metadata,
        )


def _flatten_tensors(value: Any) -> Iterable[Any]:
    """Yield Tensor leaves from nested model arguments or outputs. [基础设施]

    Args:
        value: Any — Tensor, mapping, sequence, or scalar container.
    Yields:
        Any — each Tensor leaf in deterministic container order.
    """
    import torch

    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_tensors(item)
