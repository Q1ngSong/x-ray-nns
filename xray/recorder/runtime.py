"""Small recording facade: the Torch recorder collects its graph and events through it."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from xray.ir import EventRecord, InferenceTrace, OperationRecord, TensorRecord


class TraceRecorder:
    """Collect graph records and runtime events in execution order. [主线]"""

    def __init__(self, model_name: str, *, trace_id: str | None = None) -> None:
        self._trace = InferenceTrace(
            model_name=model_name,
            trace_id=trace_id or f"trace-{uuid4().hex[:12]}",
        )
        self._next_step = 0

    def add_operation(self, operation: OperationRecord) -> None:
        """Register a computation node before it appears in an event. [主线]

        Args:
            operation: OperationRecord — node with Tensor input/output IDs.
        """
        self._trace.operations.append(operation)

    def add_tensor(self, tensor: TensorRecord) -> None:
        """Register a Tensor edge and its metadata. [主线]

        Args:
            tensor: TensorRecord — edge metadata; raw values remain external.
        """
        self._trace.tensors.append(tensor)

    def emit(self, event_type: str, *, tensor: str | None = None, operation: str | None = None, **payload: Any) -> EventRecord:
        """Append one event using the recorder's monotonic runtime step. [主线]

        Args:
            event_type: str — event name such as `consume_tensor` or `execute_operation`.
            tensor: str | None — referenced Tensor ID when applicable.
            operation: str | None — referenced Operation ID when applicable.
            payload: Any — small JSON-compatible event details.
        Returns:
            EventRecord — the event appended to the trace.
        """
        event = EventRecord(self._next_step, event_type, tensor, operation, payload)
        self._trace.events.append(event)
        self._next_step += 1
        return event

    def build(self, *, inputs: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> InferenceTrace:
        """Finalize and validate the recorded graph without copying raw Tensor data. [主线]

        Args:
            inputs: dict[str, Any] | None — serializable input descriptors.
            metadata: dict[str, Any] | None — serializable run metadata.
        Returns:
            InferenceTrace — validated trace ready for export.
        """
        self._trace.inputs = inputs or {}
        self._trace.metadata = metadata or {}
        self._trace.validate()
        return self._trace
