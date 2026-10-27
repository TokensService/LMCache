# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Generic, Optional, TypeVar, cast
import threading

# First Party
from lmcache import torch_dev
from lmcache.utils import lmcache_deprecate
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)

T = TypeVar("T")


class MessagingFuture(Generic[T]):
    def __init__(self) -> None:
        self.is_done_ = threading.Event()
        self.result_: T | None = None
        self.exception_: BaseException | None = None
        self._retained_references: list[object] = []

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        return self.is_done_.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            bool: True if the future is done, False if the timeout was reached.
        """
        return self.is_done_.wait(timeout)

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            raise LMCacheTimeoutError("Future result not available within timeout")
        if self.exception_ is not None:
            raise self.exception_
        return cast(T, self.result_)

    def set_result(self, result: T) -> None:
        """
        Set the result of the future and mark it as done. This function is NOT
        SUPPOSED TO BE CALLED by users directly. It should be only called by
        the messaging system when the result is available.

        Args:
            result (T): The result to set.
        """
        self.result_ = result
        self.is_done_.set()

    def set_exception(self, exception: BaseException) -> None:
        """Set a request exception and mark this future as complete.

        Note:
            The ZMQ client uses this method for local outbound failures before a
            request reaches the server. Asynchronous transports such as gRPC can
            also use it to propagate RPC failures.

        Args:
            exception: Failure raised while processing the request.
        """
        self.exception_ = exception
        self.is_done_.set()

    def retain_reference(self, value: object) -> None:
        """Keep a resource alive for at least the lifetime of this future.

        Async callers can use this for resources, such as exported IPC events,
        whose validity must extend until a remote operation completes.

        Args:
            value: Resource whose lifetime must be tied to this future.
        """
        self._retained_references.append(value)

    def release_references(self) -> None:
        """Drop resources previously retained on this future.

        Callers that synchronously wait for completion can use this once the
        remote side has definitively finished with the exported resources.
        """
        self._retained_references.clear()

    def to_device_future(
        self,
        device: Any | None = None,
        event_backend: EventIPCBackend | None = None,
    ) -> "DeviceMessagingFuture":
        """Wrap this future in a device-aware future.

        Args:
            device: The device whose event backend orders completion. Defaults
                to the active device.
            event_backend: Backend already selected and validated by the
                caller during initialization. When omitted, it is resolved for
                backward compatibility.

        Returns:
            A DeviceMessagingFuture pending on both this future and the event.
        """
        # TODO: need extra type checking for the future type
        return DeviceMessagingFuture.FromMessagingFuture(
            self,  # type: ignore[arg-type]
            device,
            event_backend,
        )

    @lmcache_deprecate("Use to_device_future() instead")
    def to_cuda_future(
        self,
        device: Any | None = None,
    ) -> "DeviceMessagingFuture[T]":
        """Return a device-aware future using the deprecated CUDA name.

        Args:
            device: Device on which the completion event will be imported.

        Returns:
            A device-aware future wrapping this messaging future.
        """
        return self.to_device_future(device)


class DeviceMessagingFuture(MessagingFuture[T]):
    """
    The future class that wraps both a result and one or more device IPC events.
    The `query`, `wait`, and `result` methods pend on both the original
    future and the device event(s), ordered through the platform event backend.
    The original future should return tuple[bytes | list[bytes], T], where the
    first element is the serialized device event handle (a list of handles for
    per-layer retrieves). An empty handle means the remote side submitted no
    device work, so completion of the original future is also terminal
    completion of this future.
    """

    def __init__(
        self,
        raw_future: MessagingFuture[tuple[bytes | list[bytes], T]],
        device: Any | None = None,
        event_backend: EventIPCBackend | None = None,
    ) -> None:
        super().__init__()
        self.raw_future_ = raw_future
        self.event_: Any | None = None
        self.events_: list[Any] = []
        self.result_: T | None = None
        self._raw_response_processed = False
        self.device_ = device if device is not None else torch_dev.current_device()
        if event_backend is None:
            event_backend = get_event_ipc_backend(self.device_)
            event_backend.check_event_support(self.device_)
        self._event_backend = event_backend

    def _on_raw_future_complete(self) -> None:
        """
        Update the device event and result when the raw future is complete.
        """
        if self._raw_response_processed:
            return

        event_bytes, result = self.raw_future_.result()
        events: list[Any] = []
        if event_bytes:
            if isinstance(event_bytes, (list, tuple)):
                # Per-layer retrieve: one handle per layer, in layer order.
                events = [
                    self._event_backend.import_event(handle, self.device_)
                    for handle in event_bytes
                ]
            else:
                events = [
                    self._event_backend.import_event(event_bytes, self.device_)
                ]

        self.result_ = result
        self.events_ = events
        # Backward-compatible view: the first (or only) event. The whole
        # retrieve completes only when every event in ``events_`` has fired.
        self.event_ = events[0] if events else None
        self._raw_response_processed = True

    def ensure_resolved(self, timeout: Optional[float] = None) -> None:
        """Block until the raw response has arrived and events are imported.

        This waits only for the remote handler to *return* (the work is already
        enqueued on the server stream); it does not synchronize the device, so
        it is cheap enough to call from ``wait_for_layer_load`` inside forward.

        Args:
            timeout: Optional timeout for the underlying raw future.

        Raises:
            LMCacheTimeoutError: If the raw future does not complete in time.
        """
        if self._raw_response_processed:
            return
        flag = self.raw_future_.wait(timeout)
        if not flag:
            raise LMCacheTimeoutError(
                "DeviceMessagingFuture raw response not available within timeout"
            )
        self._on_raw_future_complete()

    def wait_for_layer(
        self,
        layer_idx: int,
        stream: Any | None = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Wait (GPU-side) for one layer's device event on ``stream``.

        Used by layer-wise loading: the current forward stream waits for the
        server's per-layer H2D event so attention for ``layer_idx`` can start
        as soon as that layer's KV is resident, without blocking on later
        layers. This is a stream operation (``cudaStreamWaitEvent``), not a
        CPU block, so it is compatible with CUDA graph capture.

        Args:
            layer_idx: Local (PP-stage) layer index.
            stream: Target stream (defaults to the current stream).
            timeout: Optional timeout for resolving the raw response (the
                server returns as soon as the copies are enqueued, so a
                missing response means a protocol/deployment mismatch, e.g.
                an old server that does not know the layerwise request).

        Returns:
            True if the wait was issued, False when this future carries no
            device work for that layer (nothing to wait on).

        Raises:
            LMCacheTimeoutError: If the raw response does not arrive within
                ``timeout``.
            IndexError: If the response carried fewer per-layer events than
                ``layer_idx`` (protocol/server mismatch).
        """
        if not self._raw_response_processed:
            self.ensure_resolved(timeout)
        if layer_idx >= len(self.events_):
            raise IndexError(
                f"wait_for_layer({layer_idx}) but the retrieve response only "
                f"carries {len(self.events_)} per-layer events; the server "
                "reported fewer layers than the worker is waiting on."
            )
        evt = self.events_[layer_idx]
        if evt is None:
            return False
        if stream is None:
            stream = torch_dev.current_stream()
        self._event_backend.wait_event(evt, stream)
        return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done, ordered through the device event(s).

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the device events. (NOTE: this could be improved
                with careful threading management)

        Returns:
            bool: True if the future is done, False if the timeout was reached.

        Notes:
            This function does not support waiting for a specific time.
            For per-layer responses, every imported event is synchronized, so
            completion means the *whole* retrieve finished.
        """
        if self._raw_response_processed:
            for evt in self.events_:
                if evt is not None:
                    self._event_backend.synchronize_event(evt, self.device_)
            return True

        flag = self.raw_future_.wait(timeout)
        if not flag:
            return False

        self._on_raw_future_complete()

        for evt in self.events_:
            if evt is not None:
                self._event_backend.synchronize_event(evt, self.device_)

        return True

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the device events. (NOTE: this could be improved
                with careful threading management)

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            raise LMCacheTimeoutError(
                "DeviceMessagingFuture result not available within timeout"
            )

        assert self.result_ is not None
        return self.result_

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        if self._raw_response_processed:
            if not self.events_:
                return True
            return all(
                self._event_backend.query_event(evt)
                for evt in self.events_
                if evt is not None
            )

        if self.raw_future_.query():
            self._on_raw_future_complete()
            if not self.events_:
                return True
            return all(
                self._event_backend.query_event(evt)
                for evt in self.events_
                if evt is not None
            )

        return False

    def set_result(self, result: T) -> None:
        raise NotImplementedError(
            "DeviceMessagingFuture does not support set_result directly"
        )

    @staticmethod
    def FromMessagingFuture(
        raw_future: MessagingFuture[tuple[bytes, T]],
        device: Any | None = None,
        event_backend: EventIPCBackend | None = None,
    ) -> "DeviceMessagingFuture[T]":
        return DeviceMessagingFuture(raw_future, device, event_backend)


# Backward-compatible alias for existing imports.
CUDAMessagingFuture = DeviceMessagingFuture
