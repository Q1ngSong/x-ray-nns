"""Data model for the Operation/Tensor/Event inference trace IR."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TensorStats:
    """Tensor statistics used by generic renderers. [基础设施]"""

    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    norm: float | None = None
    sparsity: float | None = None
    nan_count: int = 0
    inf_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize statistics with stable JSON field names. [基础设施]

        Returns:
            dict[str, Any] — JSON-compatible scalar statistics.
        """
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "norm": self.norm,
            "sparsity": self.sparsity,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


@dataclass(frozen=True)
class TensorStorage:
    """Describe optional preview and raw Tensor payloads. [基础设施]"""

    kind: str = "metadata"
    path: str | None = None
    preview_path: str | None = None
    full: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize storage metadata for a bundle manifest. [基础设施]

        Returns:
            dict[str, Any] — storage kind and relative bundle paths.
        """
        return {
            "kind": self.kind,
            "path": self.path,
            "preview_path": self.preview_path,
            "full": self.full,
        }


@dataclass(frozen=True)
class TensorRecord:
    """Represent one Tensor as a first-class graph edge. [主线]"""

    id: str
    shape: tuple[int, ...]
    dtype: str
    device: str = "cpu"
    producer: str | None = None
    consumers: tuple[str, ...] = ()
    semantic_type: str | None = None
    stats: TensorStats | None = None
    storage: TensorStorage | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize a Tensor without embedding raw values. [主线]

        Returns:
            dict[str, Any] — stable IR representation for `trace.json`.
        """
        return {
            "id": self.id,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "producer": self.producer,
            "consumers": list(self.consumers),
            "semantic_type": self.semantic_type,
            "stats": self.stats.to_dict() if self.stats else None,
            "storage": self.storage.to_dict() if self.storage else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OperationRecord:
    """Represent a computation node that consumes and produces Tensors. [主线]"""

    id: str
    type: str
    name: str
    module: str | None = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    level: str = "operator"
    source: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize an Operation with explicit Tensor references. [主线]

        Returns:
            dict[str, Any] — stable IR representation for `trace.json`.
        """
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "module": self.module,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "level": self.level,
            "source": dict(self.source) if self.source else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EventRecord:
    """Represent one ordered runtime action in the playback timeline. [主线]"""

    step: int
    type: str
    tensor: str | None = None
    operation: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize a runtime event while preserving its step. [主线]

        Returns:
            dict[str, Any] — timeline event for `trace.json`.
        """
        return {
            "step": self.step,
            "type": self.type,
            "tensor": self.tensor,
            "operation": self.operation,
            "payload": dict(self.payload),
        }


@dataclass
class InferenceTrace:
    """Bundle the graph and runtime event stream for one inference. [主线]"""

    model_name: str
    operations: list[OperationRecord] = field(default_factory=list)
    tensors: list[TensorRecord] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    trace_id: str = "trace"
    schema_version: str = "0.1"
    inputs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Check IDs, references, and monotonic runtime order. [基础设施]

        Raises:
            ValueError — if the trace contains duplicate IDs or dangling references.
        """
        operation_ids = {item.id for item in self.operations}
        tensor_ids = {item.id for item in self.tensors}
        if len(operation_ids) != len(self.operations):
            raise ValueError("operation IDs must be unique")
        if len(tensor_ids) != len(self.tensors):
            raise ValueError("tensor IDs must be unique")
        for operation in self.operations:
            missing = (set(operation.inputs) | set(operation.outputs)) - tensor_ids
            if missing:
                raise ValueError(f"operation {operation.id} references unknown tensors: {sorted(missing)}")
            for tensor_id in operation.outputs:
                tensor = next(item for item in self.tensors if item.id == tensor_id)
                if tensor.producer != operation.id:
                    raise ValueError(f"tensor {tensor_id} producer does not match operation {operation.id}")
            for tensor_id in operation.inputs:
                tensor = next(item for item in self.tensors if item.id == tensor_id)
                if operation.id not in tensor.consumers:
                    raise ValueError(f"tensor {tensor_id} consumers do not include operation {operation.id}")
        for tensor in self.tensors:
            references = set(filter(None, (tensor.producer,))) | set(tensor.consumers)
            missing = references - operation_ids
            if missing:
                raise ValueError(f"tensor {tensor.id} references unknown operations: {sorted(missing)}")
        previous_step = -1
        for event in self.events:
            if event.step < 0 or event.step <= previous_step:
                raise ValueError("events must be ordered by strictly increasing non-negative runtime step")
            if event.tensor is not None and event.tensor not in tensor_ids:
                raise ValueError(f"event references unknown tensor: {event.tensor}")
            if event.operation is not None and event.operation not in operation_ids:
                raise ValueError(f"event references unknown operation: {event.operation}")
            previous_step = event.step

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete trace manifest. [主线]

        Returns:
            dict[str, Any] — JSON-compatible graph, tensors, and event stream.
        """
        self.validate()
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "model_name": self.model_name,
            "inputs": dict(self.inputs),
            "operations": [item.to_dict() for item in self.operations],
            "tensors": [item.to_dict() for item in self.tensors],
            "events": [item.to_dict() for item in self.events],
            "metadata": dict(self.metadata),
        }

    def save_json(self, path: str | Path) -> Path:
        """Write a validated trace manifest to a UTF-8 JSON file. [基础设施]

        Args:
            path: str | Path — destination file; parent directories are created.
        Returns:
            Path — resolved destination path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load_json(cls, path: str | Path) -> "InferenceTrace":
        """Read a trace manifest such as a bundle's ``trace.json`` back into records. [基础设施]

        Args:
            path: str | Path — UTF-8 JSON written by ``save_json`` or ``export_bundle``;
                dangling references or out-of-order events raise ``ValueError`` from ``validate``.
        Returns:
            InferenceTrace — trace whose ``to_dict()`` equals the file content.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        stat_names = {"min": "minimum", "max": "maximum"}
        tensors = [
            TensorRecord(
                item["id"], tuple(item["shape"]), item["dtype"], item.get("device", "cpu"), item.get("producer"),
                tuple(item.get("consumers", ())), item.get("semantic_type"),
                TensorStats(**{stat_names.get(key, key): value for key, value in item["stats"].items()}) if item.get("stats") else None,
                TensorStorage(**item["storage"]) if item.get("storage") else None, item.get("metadata") or {},
            )
            for item in data["tensors"]
        ]
        operations = [
            OperationRecord(
                item["id"], item["type"], item["name"], item.get("module"), tuple(item.get("inputs", ())),
                tuple(item.get("outputs", ())), item.get("level", "operator"), item.get("source"), item.get("metadata") or {},
            )
            for item in data["operations"]
        ]
        events = [
            EventRecord(item["step"], item["type"], item.get("tensor"), item.get("operation"), item.get("payload") or {})
            for item in data["events"]
        ]
        trace = cls(data["model_name"], operations, tensors, events, data.get("trace_id", "trace"),
                    data.get("schema_version", "0.1"), data.get("inputs") or {}, data.get("metadata") or {})
        trace.validate()
        return trace
