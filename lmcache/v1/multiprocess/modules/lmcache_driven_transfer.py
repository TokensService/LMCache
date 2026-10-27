# SPDX-License-Identifier: Apache-2.0
"""LMCache-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass
from itertools import islice
from typing import Any, Generator, Sequence
import threading
import time

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.gpu_connector.gpu_ops import (
    build_staging_copies,
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType, next_transfer_key
from lmcache.v1.mp_observability.event_bus import (
    get_event_bus,
    is_observability_enabled,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lookup import resolve_prefetched_obj_keys
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.cache_context import create_cache_context
from lmcache.v1.platform.torch_ops import (
    _per_layer_paged_shape,
    _tensor_from_ptr,
)
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)
_HAS_NATIVE_OBJECT_GROUP_TRANSFER: bool = hasattr(
    device_ops, "execute_object_group_transfer"
)
_HAS_TRANSFER_PHASE_TIMING: bool = hasattr(device_ops, "pop_completed_phase_timings")


def get_layout_desc(
    cache_context: BaseCacheContext,
    num_tokens: int,
    object_group_id: int,
) -> MemoryLayoutDesc:
    """Get the memory layout description for a specific object group.

    The returned layout describes the single memory object that backs
    ``object_group_id``: one (shape, dtype) entry per kernel group in that
    object group, in the kernel groups' declared layout order. Kernel groups
    may have different shapes and dtypes.

    Args:
        cache_context: The cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.
        object_group_id: Index of the object group whose layout to build.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and
        dtypes, one entry per kernel group in the object group.
    """
    object_group = cache_context.kv_layer_groups_manager.object_groups[object_group_id]
    shapes_and_dtypes = [
        cache_context.get_kernel_group_shape_dtype(num_tokens, kernel_group_idx)
        for kernel_group_idx in object_group.kernel_group_indices
    ]
    shapes, dtypes = zip(*shapes_and_dtypes, strict=False)
    return MemoryLayoutDesc(shapes=list(shapes), dtypes=list(dtypes))


def batched_iteration_with_skip(
    lst: Sequence,
    batch_size: int,
    skip_count: int,
) -> Generator[tuple[int, tuple], None, None]:
    """Utility function to iterate over a list in batches with an initial skip.

    Args:
        lst: The list to iterate over.
        batch_size: The size of each batch.
        skip_count: The number of items to skip at the start of the list.

    Yields:
        Tuples of (batch_start_idx, batch) where batch is a tuple of items
        from the list, and batch_start_idx is the "original" index of the first
        item in the batch.

    Raises:
        ValueError: If batch_size is less than 1 or skip_count is negative.

    Note:
        Batch_idx is the index of the batch in the original list, accounting
        for the skipped items. For example, if skip_count is 10 and batch_size
        is 5, the first yielded batch will have batch_start_idx=10.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    if skip_count < 0:
        raise ValueError("skip_count must be non-negative")

    it = iter(lst)
    # Skip the initial items
    for _ in range(skip_count):
        next(it, None)
    batch_start_idx = skip_count
    while batch := tuple(islice(it, batch_size)):
        yield batch_start_idx, batch
        batch_start_idx += len(batch)


def all_null_chunk_masks(
    block_ids: Sequence[Sequence[int]],
    object_groups: Sequence[ObjectGroupInfo],
    blocks_per_chunk: Sequence[int],
    num_chunks: int,
) -> list[list[bool]]:
    """Mark, per object group, the chunks whose engine block ids are all null.

    A chunk is null for an object group when every block id of every kernel
    group in that group is 0 (the vLLM null block). Align-mode Mamba/linear
    layers produce such chunks: only the block holding the last recurrent state
    is real, so every earlier chunk is null. These chunks must not be stored --
    the null block carries no valid KV, and object keys are content hashes, so
    committing them would serve garbage to a later prefix hit.

    Args:
        block_ids: Raw per-kernel-group engine block ids (before any downsample),
            indexed by kernel-group index.
        object_groups: The object groups, indexed by object-group id.
        blocks_per_chunk: Blocks in one chunk per kernel group, indexed by
            kernel-group index.
        num_chunks: Number of chunks in the request.

    Returns:
        ``mask[g][i]`` is True iff chunk ``i`` is all-null for object group ``g``.
    """
    masks: list[list[bool]] = []
    for group in object_groups:
        chunk_null: list[bool] = []
        for i in range(num_chunks):
            is_null = True
            for kg in group.kernel_group_indices:
                bpc = blocks_per_chunk[kg]
                if any(block_ids[kg][i * bpc : (i + 1) * bpc]):
                    is_null = False
                    break
            chunk_null.append(is_null)
        masks.append(chunk_null)
    return masks


def downsample_and_stage_block_ids(
    cache_context: BaseCacheContext,
    block_ids: list[list[int]],
) -> list[torch.Tensor]:
    """Cut the block id lists to skip the unneeded blocks in a chunk and
    stage it into GPU tensors for later use.

    This mainly targets the case where a portion of the blocks are not
    needed for every chunk, such as deepseek v4's swa cache.

    Note that the we do NOT do any object-level skipping here.

    Args:
        cache_context: The cache context containing the KV cache information.
        block_ids: The original block id lists, indexed by LMCache KV group index.

    Returns:
        The cut block id lists, indexed by LMCache KV group index.

    Note:
        This function has some coupled logic with transfer_kv_per_object_group below.
        The caller need to make sure that the block ids seen by
        transfer_kv_per_object_group are produced by this function.

    Example:
        If a model have 2 kernel groups, one is full attention with block size 32,
        one is swa attention with block size 32 and sliding window size 64, and
        LMCache has a chunk size of 128. And there are 2 chunks in total (256 tokens).

        The input will be:
        [
          [1, 2, 3, 4, 5, 6, 7, 8],  # block ids for the full attention group
          [11, 12, 13, 14, 15, 16, 17, 18], # block ids for the swa attention group
        ]

        The output will be
        [
          [1, 2, 3, 4, 5, 6, 7, 8],  # full attention group still needs all block ids
          [13, 14, 17, 18], # swa attention group only needs the last 2 block per chunk
        ]
    """
    num_kernel_groups = cache_context.kv_layer_groups_manager.num_kernel_groups
    for kernel_group_id in range(num_kernel_groups):
        subchunk_sw_size_tokens = (
            cache_context.kv_layer_groups_manager.get_subchunk_sw_size_tokens(
                kernel_group_id
            )
        )
        tokens_per_chunk = min(
            cache_context.lmcache_tokens_per_chunk, subchunk_sw_size_tokens
        )
        keep_blocks_per_chunk = cache_context.calculate_num_blocks(
            tokens_per_chunk, kernel_group_id
        )
        total_blocks_per_chunk = cache_context.calculate_num_blocks(
            cache_context.lmcache_tokens_per_chunk, kernel_group_id
        )

        new_block_ids = []
        old_block_ids = block_ids[kernel_group_id]
        assert len(old_block_ids) % total_blocks_per_chunk == 0, (
            f"len(block_ids[{kernel_group_id}]) should be a multiple "
            f"of total_blocks_per_chunk ({total_blocks_per_chunk}), but got "
            f"{len(old_block_ids)}"
        )

        for i in range(0, len(old_block_ids), total_blocks_per_chunk):
            chunk_block_ids = old_block_ids[i : i + total_blocks_per_chunk]
            new_block_ids.extend(chunk_block_ids[-keep_blocks_per_chunk:])

        block_ids[kernel_group_id] = new_block_ids

    # Stage the cut block ids into GPU tensors
    block_ids_gpu = cache_context.stage_block_ids(block_ids)
    return block_ids_gpu


def _recalculate_blocks_to_skip(
    blocks_per_chunk: int,
    blocks_per_window: int,
    blocks_to_skip: int,
) -> int:
    """Re-calculate the number of blocks to skip for a batch of chunks based
    on the blocks per chunk and blocks per sliding window WHEN the window
    size is smaller than the lmcache chunk size.

    Args:
        blocks_per_chunk: The total number of blocks in one chunk for the
            current group.
        blocks_per_window: The number of blocks in the sliding window
            for the current group. Should be less than or equal to
            blocks_per_chunk.
        blocks_to_skip: The number of blocks to skip.

    Returns:
        The re-calculated number of blocks to skip for the current batch of
        chunks.
    """
    if blocks_per_chunk == blocks_per_window:
        return blocks_to_skip

    full_windows_to_skip = blocks_to_skip // blocks_per_chunk
    tail_blocks = blocks_to_skip % blocks_per_chunk
    tail_blocks_to_skip = tail_blocks - (blocks_per_chunk - blocks_per_window)
    return full_windows_to_skip * blocks_per_window + max(0, tail_blocks_to_skip)


def _run_object_group_transfer_plan(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
    *,
    transfer_key: str,
) -> None:
    """Plan and execute one object group's transfer in a single native call.

    This is the fast path of :func:`transfer_kv_per_object_group`: it runs the
    same batched-iteration / skip logic, but instead of issuing each staging
    copy and kernel launch immediately (each a GIL release/re-acquire), it
    resolves every argument to plain pointers/scalars (the "planner", GIL held
    throughout) and hands the whole plan to ``execute_object_group_transfer``,
    which issues all of it on the stream within a single GIL release.

    Requires every object to be non-GDS (staged through the lazy-allocator
    path); the caller skips groups that contain any GDS-backed object.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        block_ids_gpu: GPU block IDs, indexed by LMCache KV group index.
        memory_objs: The MemoryObj instances to copy. None entries are only
            valid for D2H (the batch is skipped); H2D raises.
        object_group_id: Index of the object group being copied.
        batch_size: Number of memory objects per batched copy.
        skip_first_n_tokens: Tokens to skip writing at the start of the range.
        direction: H2D (retrieve) or D2H (store).
        transfer_key: Identity of this store/retrieve operation, echoed back
            on every phase-timing sample (a request issues several transfers,
            so the request id cannot identify one).

    Raises:
        ValueError: If a None entry is found in memory_objs when direction is
            H2D, or if an object's size does not match its GPU staging buffer.
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    object_group = kv_groups_manager.object_groups[object_group_id]
    kernel_group_ids = object_group.kernel_group_indices
    is_h2d = direction == lmcache_native.TransferDirection.H2D
    max_batch_size = cache_context.max_batch_size

    # --- Per-kernel-group invariants, resolved once (vs. every batch before) ---
    kernel_group_specs: list[Any] = []
    spec_index_by_kg: dict[int, int] = {}
    blocks_per_chunk_by_kg: dict[int, int] = {}
    blocks_per_window_by_kg: dict[int, int] = {}
    for kernel_group_id in kernel_group_ids:
        blocks_per_chunk = cache_context.calculate_num_blocks(
            lmcache_chunk_size, kernel_group_id
        )
        tokens_per_window = min(
            lmcache_chunk_size,
            kv_groups_manager.get_subchunk_sw_size_tokens(kernel_group_id),
        )
        blocks_per_window = cache_context.calculate_num_blocks(
            tokens_per_window, kernel_group_id
        )
        blocks_per_chunk_by_kg[kernel_group_id] = blocks_per_chunk
        blocks_per_window_by_kg[kernel_group_id] = blocks_per_window

        paged_ptrs = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
        block_ids_tensor = block_ids_gpu[kernel_group_id]
        temp_buffers = [
            cache_context.get_temp_kernel_group_buffer(slot, kernel_group_id)
            for slot in range(max_batch_size)
        ]

        spec_index_by_kg[kernel_group_id] = len(kernel_group_specs)
        kernel_group_specs.append(
            device_ops.KernelGroupSpec(
                paged_ptrs.data_ptr(),
                [buffer.data_ptr() for buffer in temp_buffers],
                cache_context.get_shape_desc(kernel_group_id),
                cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
                cache_context.get_engine_kv_format(kernel_group_id),
                block_ids_tensor.data_ptr(),
                block_ids_tensor.numel(),
            )
        )

    # Temp object-group staging buffers (reused per batch slot, like above).
    object_group_buffers = [
        cache_context.get_temp_object_group_buffer(slot, object_group_id)
        for slot in range(max_batch_size)
    ]

    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = 0
    if not attn_desc.is_full_attention(object_group_id) and is_h2d:
        sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
        num_objects_to_skip = max(0, len(memory_objs) - sw_size_chunks)
        logger.debug(
            "Detected sliding window for object group %d: "
            "skipping the first %d objects in the batch",
            object_group_id,
            num_objects_to_skip,
        )

    # --- Walk the batches in order, emitting staging + launch work per step ---
    batch_steps: list[Any] = []
    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs, batch_size, skip_count=num_objects_to_skip
    ):
        if any(mo is None for mo in memory_object_batch):
            if is_h2d:
                raise ValueError(
                    "MemoryObj is None for some objects in the batch, cannot "
                    "perform H2D copy. memory_object_batch: "
                    f"{memory_object_batch}"
                )
            else:
                continue

        batch_len = len(memory_object_batch)
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size

        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue

        skip_tokens_in_chunk = effective_start - batch_start_token

        staging = build_staging_copies(
            memory_object_batch,
            object_group_buffers[:batch_len],
            is_h2d,
        )

        launches: list[Any] = []
        for kernel_group_id in kernel_group_ids:
            blocks_per_chunk = blocks_per_chunk_by_kg[kernel_group_id]
            blocks_per_window = blocks_per_window_by_kg[kernel_group_id]

            start_block_pos = start_object_idx * blocks_per_window
            end_block_pos = (start_object_idx + batch_len) * blocks_per_window

            orig_skip_blocks = cache_context.calculate_num_blocks(
                skip_tokens_in_chunk, kernel_group_id
            )
            recalculated_skip_blocks = _recalculate_blocks_to_skip(
                blocks_per_chunk,
                blocks_per_window,
                orig_skip_blocks,
            )

            launches.append(
                device_ops.LaunchVar(
                    spec_index_by_kg[kernel_group_id],
                    start_block_pos,
                    end_block_pos - start_block_pos,
                    batch_len,
                    recalculated_skip_blocks,
                )
            )

        batch_steps.append(device_ops.BatchStep(staging, launches))

    if not batch_steps:
        return

    # Time the phases only when a subscriber consumes the samples. An older
    # compiled extension has neither the keywords nor anything to consume
    # them, so fall back to the untimed legacy signature.
    timing_kwargs = (
        {
            "phase_timing_enabled": is_observability_enabled()
            and get_event_bus().has_subscribers(EventType.MP_TRANSFER_PHASE_SAMPLES),
            # Echoed back verbatim on each sample; the transfer's identity.
            "session_id": transfer_key,
        }
        if _HAS_TRANSFER_PHASE_TIMING
        else {}
    )
    device_ops.execute_object_group_transfer(
        direction,
        cache_context.device,
        LazyMemoryAllocator.PIN_CHUNK_SIZE,
        kernel_group_specs,
        batch_steps,
        **timing_kwargs,
    )


def transfer_kv_per_object_group(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
    *,
    transfer_key: str,
) -> None:
    """Helper function to transfer memory objects of a single object group
    to/from GPU, with batching support.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        block_ids_gpu: GPU block IDs to retrieve into, indexed by LMCache KV group
            index. It should satisfy `len(block_ids_gpu[i]) == len(memory_objs) *
            blocks_per_chunk[i]` for each group `i`.
            Note that the block IDs list are already on GPU.
        memory_objs: The list of MemoryObj instances to copy from. It could be
            None when allocation or retrieval fails. For store (D2H), it should
            ignore the None entry and continue copying the rest. For retrieve
            (H2D), it should raise the error and stop copying.
        object_group_id: Index of the object group being copied.
        batch_size: The number of memory objects to perform batched copy
        skip_first_n_tokens: Number of tokens to skip writing at the start of
            the retrieve range. This avoids overwriting APC-shared GPU blocks that
            may be read concurrently by other requests.
        direction: The transfer direction, H2D (retrieve) or D2H (store).
        transfer_key: Identity of this store/retrieve operation, echoed back on
            every phase-timing sample; see _run_object_group_transfer_plan.

    Raises:
        ValueError: If it founds None entry in memory_objs when direction is H2D.
    Note:
        This function expects the caller to stage the block ids (list[list[int]])
        into GPU tensors and pass them in as `block_ids_gpu`.
    """
    if _HAS_NATIVE_OBJECT_GROUP_TRANSFER and not any(
        isinstance(mo, GDSMemoryObject) for mo in memory_objs
    ):
        _run_object_group_transfer_plan(
            cache_context,
            block_ids_gpu,
            memory_objs,
            object_group_id,
            batch_size,
            skip_first_n_tokens,
            direction,
            transfer_key=transfer_key,
        )
        return

    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    object_group = kv_groups_manager.object_groups[object_group_id]
    kernel_group_ids = object_group.kernel_group_indices
    is_h2d = direction == lmcache_native.TransferDirection.H2D

    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = 0
    if not attn_desc.is_full_attention(object_group_id) and is_h2d:
        sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
        num_objects_to_skip = max(0, len(memory_objs) - sw_size_chunks)
        logger.debug(
            "Detected sliding window for object group %d: "
            "skipping the first %d objects in the batch",
            object_group_id,
            num_objects_to_skip,
        )

    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs, batch_size, skip_count=num_objects_to_skip
    ):
        if any(mo is None for mo in memory_object_batch):
            if is_h2d:
                raise ValueError(
                    "MemoryObj is None for some objects in the batch, cannot "
                    "perform H2D copy. memory_object_batch: "
                    f"{memory_object_batch}"
                )
            else:
                continue

        batch_len = len(memory_object_batch)
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size

        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue

        skip_tokens_in_chunk = effective_start - batch_start_token

        # For H2D, copy from CPU to GPU tmp buffers before the kernel launch
        if is_h2d:
            for chunk_idx, memory_obj in enumerate(memory_object_batch):
                lmcache_memcpy_async_h2d(
                    memory_obj,
                    cache_context.get_temp_object_group_buffer(
                        chunk_idx, object_group_id
                    ),
                )

        # Do paged KV copy
        for kernel_group_id in kernel_group_ids:
            blocks_per_chunk = cache_context.calculate_num_blocks(
                lmcache_chunk_size, kernel_group_id
            )
            tokens_per_window = min(
                lmcache_chunk_size,
                kv_groups_manager.get_subchunk_sw_size_tokens(kernel_group_id),
            )
            blocks_per_window = cache_context.calculate_num_blocks(
                tokens_per_window, kernel_group_id
            )

            # Get the block ids for this chunk
            start_block_pos = start_object_idx * blocks_per_window
            end_block_pos = (start_object_idx + batch_len) * blocks_per_window

            block_ids_curr_batch = block_ids_gpu[kernel_group_id][
                start_block_pos:end_block_pos
            ]

            # Re-calculate the skip blocks for this kernel group
            orig_skip_blocks = cache_context.calculate_num_blocks(
                skip_tokens_in_chunk, kernel_group_id
            )
            recalculated_skip_blocks = _recalculate_blocks_to_skip(
                blocks_per_chunk,
                blocks_per_window,
                orig_skip_blocks,
            )

            # Launch kernel
            group_kv_pointers = cache_context.get_kernel_group_kv_pointers(
                kernel_group_id
            )
            group_lmcache_chunk_size = cache_context.get_slots_per_chunk_in_sw(
                kernel_group_id
            )
            tmp_gpu_buffers_batched = [
                cache_context.get_temp_kernel_group_buffer(
                    i, kernel_group_id
                ).data_ptr()
                for i in range(batch_len)
            ]
            device_ops.multi_layer_block_kv_transfer(
                group_kv_pointers,
                tmp_gpu_buffers_batched,
                block_ids_curr_batch,
                cache_context.device,
                direction,
                cache_context.get_shape_desc(kernel_group_id),
                group_lmcache_chunk_size,
                cache_context.get_engine_kv_format(kernel_group_id),
                recalculated_skip_blocks,
            )

        # For D2H, copy from GPU tmp buffers to CPU after the kernel launch
        if not is_h2d:
            for chunk_idx, memory_obj in enumerate(memory_object_batch):
                lmcache_memcpy_async_d2h(
                    cache_context.get_temp_object_group_buffer(
                        chunk_idx, object_group_id
                    ),
                    memory_obj,
                )


def _object_group_layerwise_supported(
    cache_context: BaseCacheContext,
    object_group_id: int,
) -> bool:
    """Return True when the per-layer transfer path can serve this object group.

    The per-layer path slices each chunk's KV_2LTD blob (``[2, num_layers,
    num_tokens, hidden]``) along its layer axis and issues one
    ``single_layer_kv_transfer`` per layer with a per-token slot mapping, so
    it requires:

    - exactly one kernel group per object group (the merged hidden-dim layout
      of hybrid groups is not sliced here),
    - a per-layer paged KV format (not cross-layer / SGLang 2-plane / MLA),
    - full attention (no sliding-window chunk skipping; the ``skip_first_n_``
      token prefix is expressed via ``-1`` entries in the slot mapping).

    GDS-backed objects are probed separately (they are uniform per L1 backend,
    so the first read group decides for the whole retrieve).

    Args:
        cache_context: The GPU cache context containing the KV cache.
        object_group_id: Index of the object group being copied.

    Returns:
        True when the object group can be transferred per layer.
    """
    object_group = cache_context.kv_layer_groups_manager.object_groups[object_group_id]
    if len(object_group.kernel_group_indices) != 1:
        return False
    kg_id = object_group.kernel_group_indices[0]
    engine_kv_format = cache_context.get_engine_kv_format(kg_id)
    if (
        lmcache_native.is_cross_layer(engine_kv_format)
        or lmcache_native.is_kv_list(engine_kv_format)
        or lmcache_native.is_mla(engine_kv_format)
    ):
        return False
    attn_desc = cache_context.kv_layer_groups_manager.get_attn_desc()
    if not attn_desc.is_full_attention(object_group_id):
        return False
    return True


def _memory_obj_layer_tensor(
    memory_obj: MemoryObj,
    layer_idx: int,
    num_layers: int,
    full_shape: torch.Size,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a zero-copy view of one layer of a KV_2LTD memory object.

    ``full_shape`` is the object's logical shape ``[2, num_layers, num_tokens,
    hidden]`` (single kernel-group object groups only; see
    :func:`_object_group_layerwise_supported`). Tensor-backed objects are
    sliced directly; lazy-allocator objects (``raw_tensor is None``) get an
    ``as_strided`` view over their raw pointer with the same strides, so both
    cases hand ``single_layer_kv_transfer`` a ``[2, num_tokens, hidden]`` view
    whose K/V plane stride still encodes the full multi-layer layout.

    Args:
        memory_obj: The CPU memory object (whole chunk, all layers).
        layer_idx: Local layer index to slice out.
        num_layers: Total local layer count (the blob's layer axis length).
        full_shape: The object's logical ``[2, nl, num_tokens, hidden]`` shape.
        dtype: KV element dtype (matches the object's tensor).

    Returns:
        A ``[2, num_tokens, hidden]`` view of ``layer_idx``'s K/V planes.
    """
    num_tokens = int(full_shape[2])
    hidden = int(full_shape[3])
    raw = memory_obj.raw_tensor
    if raw is not None:
        return raw[:, layer_idx, :, :]
    ptr = int(memory_obj.data_ptr)
    base = _tensor_from_ptr(ptr, tuple(int(s) for s in full_shape), dtype, "cpu")
    layer_stride = num_layers * num_tokens * hidden
    return base.as_strided(
        (2, num_tokens, hidden),
        stride=(layer_stride, hidden, 1),
        storage_offset=layer_idx * num_tokens * hidden,
    )


def transfer_kv_per_object_group_layerwise(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    layer_idx: int,
    num_layers: int,
    *,
    transfer_key: str,
) -> None:
    """Transfer one layer of an object group from CPU into the paged GPU cache.

    This is the per-layer counterpart of
    :func:`transfer_kv_per_object_group`: instead of copying the whole chunk
    blob (all layers) in one go, it copies only ``layer_idx``'s K/V planes of
    every chunk in the batch and scatters them into that layer's paged buffer.
    Issued once per layer from the layerwise retrieve, it lets the server
    record one CUDA event per layer so the worker can overlap the forward of
    earlier layers with the H2D of later ones.

    Only supported for single-kernel-group, full-attention object groups (see
    :func:`_object_group_layerwise_supported`); the caller must not call this
    otherwise.

    Args:
        cache_context: The GPU cache context containing the KV cache.
        block_ids_gpu: GPU block IDs, indexed by LMCache KV group index.
        memory_objs: The MemoryObj instances (whole chunks) to copy from. None
            entries are not allowed (H2D).
        object_group_id: Index of the object group being copied.
        batch_size: Number of memory objects per batched copy.
        skip_first_n_tokens: Initial tokens the server must not overwrite; they
            get ``-1`` slots so the transfer skips them.
        layer_idx: Local layer index to transfer.
        num_layers: Total local layer count.
        transfer_key: Identity of this retrieve operation, echoed back on every
            phase-timing sample.

    Raises:
        ValueError: If a None entry is found in ``memory_objs``.
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    object_group = cache_context.kv_layer_groups_manager.object_groups[object_group_id]
    kernel_group_id = object_group.kernel_group_indices[0]
    shape_desc = cache_context.get_shape_desc(kernel_group_id)
    bs = int(shape_desc.bs)
    blocks_per_chunk = cache_context.calculate_num_blocks(
        lmcache_chunk_size, kernel_group_id
    )
    engine_kv_format = cache_context.get_engine_kv_format(kernel_group_id)
    full_shape, dtype = cache_context.get_kernel_group_shape_dtype(
        lmcache_chunk_size, kernel_group_id
    )
    device = cache_context.device

    # Per-layer paged destination tensor (zero-copy view over the layer's KV
    # pointers), built once for the whole transfer.
    per_layer_shape = _per_layer_paged_shape(
        engine_kv_format,
        int(shape_desc.nb),
        bs,
        int(shape_desc.nh),
        int(shape_desc.hs),
    )
    kv_ptrs = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
    if len(kv_ptrs) <= layer_idx:
        raise ValueError(
            f"Layerwise retrieve asked for layer {layer_idx} but the registered "
            f"KV cache only exposes {len(kv_ptrs)} layers"
        )
    paged_layer = _tensor_from_ptr(
        int(kv_ptrs[layer_idx].item()), per_layer_shape, dtype, device
    )

    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs, batch_size, skip_count=0
    ):
        if any(mo is None for mo in memory_object_batch):
            raise ValueError(
                "MemoryObj is None for some objects in the batch, cannot "
                "perform per-layer H2D copy. memory_object_batch: "
                f"{memory_object_batch}"
            )

        batch_len = len(memory_object_batch)
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size
        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue
        skip_tokens_in_batch = effective_start - batch_start_token

        # Per-token slot mapping for the batch: slot(t) = engine_block_id(t) *
        # block_size + t % block_size; the skipped prefix maps to -1 (invalid).
        num_tokens = batch_len * lmcache_chunk_size
        token_idx = torch.arange(num_tokens, device=device, dtype=torch.long)
        chunk_in_batch = token_idx // lmcache_chunk_size
        off_in_chunk = token_idx % lmcache_chunk_size
        flat_block_pos = (
            (start_object_idx + chunk_in_batch) * blocks_per_chunk
            + off_in_chunk // bs
        )
        bid = block_ids_gpu[kernel_group_id][flat_block_pos]
        slot = bid * bs + (off_in_chunk % bs)
        if skip_tokens_in_batch > 0:
            slot[:skip_tokens_in_batch] = -1

        for chunk_idx, memory_obj in enumerate(memory_object_batch):
            layer_src = _memory_obj_layer_tensor(
                memory_obj, layer_idx, num_layers, full_shape, dtype
            )
            chunk_slot = slot[
                chunk_idx * lmcache_chunk_size : (chunk_idx + 1) * lmcache_chunk_size
            ]
            device_ops.single_layer_kv_transfer(
                layer_src,
                paged_layer,
                chunk_slot,
                lmcache_native.TransferDirection.H2D,
                engine_kv_format,
                token_major=False,
            )


@dataclass
class ContextEntry:
    """Registered cache context metadata for a single worker instance.

    The concrete type is whatever :func:`create_cache_context` returned
    for the wrapper list at registration time -- a
    :class:`GPUCacheContext` for CUDA-IPC wrappers, a
    :class:`CPUCacheContext` for POSIX-SHM wrappers. Both expose
    the same ``kv_tensors`` / ``engine_kv_format`` / ``num_layers`` / ...
    duck-typed surface, so downstream consumers stay agnostic.

    Args:
        cache_context: Platform cache context (GPU or CPU) managing
            shape and pointers to the registered KV cache tensors.
        model_name: The name of the model associated with this KV cache.
        world_size: The world size associated with this KV cache.
        last_seen: ``time.monotonic()`` of the most recent activity from
            this instance (register, PING, store, or retrieve). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window (timeout vs registration grace).
            Latched only by PING, never by traffic.
        event_backend: Cached event backend selected for this context's device.
    """

    cache_context: BaseCacheContext
    model_name: str
    world_size: int
    last_seen: float = 0.0
    has_liveness_signal: bool = False
    event_backend: EventIPCBackend | None = None


class LMCacheDrivenTransferModule(InstanceLivenessTarget):
    """Handles LMCache-driven KV cache transfer operations.

    Owns GPU context registrations and provides handlers for
    register, unregister, store, and retrieve of GPU KV caches.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._cache_contexts: dict[int, ContextEntry] = {}
        # Guards all reads/writes of _cache_contexts. The reaper mutates it
        # off the MQ main loop, so register/unregister/store/retrieve and
        # report_status all serialize through this lock. Held only for dict
        # ops -- never across context creation, layout-registry calls, or
        # empty_cache (leaf-lock invariant: no thread holds two locks).
        self._lock = threading.Lock()

        # Route finish_write / finish_read_prefetched through a C++ host
        # callback so the driver thread doesn't acquire the GIL.
        self._device_host_func_dispatcher = DeviceHostFuncDispatcher()
        self._device_host_func_dispatcher.register(
            "finish_write",
            self._ctx.storage_manager.finish_write,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.register(
            "finish_read_prefetched",
            self._ctx.storage_manager.finish_read_prefetched,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.start()

    def register_host_func(self, kind: str, handler: Any, payload_type: Any) -> None:
        """Register *handler* for *kind* on the per-process device host-func
        dispatcher (stream-ordered callbacks without a driver-thread GIL
        acquire); pair with ``submit_callback_to_stream``."""
        self._device_host_func_dispatcher.register(kind, handler, payload_type)

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_and_touch_context_entry(self, instance_id: int) -> ContextEntry | None:
        """Return the entry for ``instance_id``, refreshing its last-seen time.

        The refresh keeps an actively transferring worker from being reaped
        even if its PINGs are briefly delayed. Does not latch the
        ping-proven flag -- only PINGs do that.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry, or None if the instance is not (or no longer) tracked.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
            return entry

    def _release_failed_retrieve_locks(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> None:
        """Release only one failed instance's unconsumed lookup locks.

        The lookup session is the ownership record.  If it is absent or does
        not match the RETRIEVE range, no release is attempted: L1 locks are
        anonymous refcounts, so guessing could consume a concurrent reader's
        lock.  ``claim_failed_retrieve_release`` also makes duplicate failure
        responses idempotent.
        """
        session = self._ctx.session_manager.get(key.request_id)
        if session is None:
            logger.warning(
                "Cannot release RETRIEVE locks for unregistered instance %d: "
                "request %s has no lookup session",
                instance_id,
                key.request_id,
            )
            return

        lock_state = session.prepare_failed_retrieve_release(key)
        if lock_state is None:
            return
        hit_chunks, locked_gids, group_windows, lookup_generation = lock_state
        obj_keys = resolve_prefetched_obj_keys(
            self._ctx,
            key,
            hit_chunks,
            locked_gids,
            group_windows=group_windows,
        )
        if not session.claim_failed_retrieve_release(
            instance_id, key, lookup_generation
        ):
            return
        if obj_keys:
            # One failed RETRIEVE owns one read lock per key.  In
            # particular, do not release the scheduler's whole MLA
            # reservation here: the remaining TP workers and concurrent
            # requests still own their independent read locks.
            self._ctx.storage_manager.finish_read_prefetched(obj_keys, read_locks=1)

    def context_entries_snapshot(self) -> dict[int, ContextEntry]:
        """Return a shallow copy of the registry for iteration or status.

        Returns:
            A new dict mapping instance ID to entry; does not refresh
            last-seen times.
        """
        with self._lock:
            return dict(self._cache_contexts)

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered instances."""
        with self._lock:
            return len(self._cache_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap GPU registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one
        that has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, ContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._cache_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                reaped.append((iid, self._cache_contexts.pop(iid)))
        reaped_ids: list[int] = []
        entries: list[ContextEntry] = []
        for iid, e in reaped:
            logger.warning(
                "Reaped GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - e.last_seen,
                e.has_liveness_signal,
            )
            reaped_ids.append(iid)
            entries.append(e)
        if reaped:
            del e  # a bound name would pin the final entry (see _release_entries)
            reaped.clear()
            self._release_entries(entries)
        return reaped_ids

    def _release_entries(self, entries: list[ContextEntry]) -> None:
        """Release a batch of entries and reclaim their device memory.

        Args:
            entries: The only remaining references to the released entries.
                The list is cleared before memory is reclaimed.
        """
        if not entries:
            return
        for entry in entries:
            entry.cache_context.close()
            self._ctx.layout_desc_registry.unregister(
                entry.model_name, entry.world_size
            )
        del entry
        entries.clear()
        # ipc_collect() only unmaps a CUDA-IPC-imported segment once its last
        # tensor reference is gone (LMCache#4014), hence the clear() above.
        torch_dev.empty_cache()
        ipc_collect = getattr(torch_dev, "ipc_collect", None)
        if ipc_collect is not None:
            # Backends without IPC collection omit this optional operation.
            ipc_collect()

    def report_status(self) -> dict:
        """Return GPU transfer module status information.

        Returns:
            A dict containing registered GPU instance IDs and
            per-instance KV cache layout metadata.
        """
        registered_gpu_ids: list[int] = []
        cache_context_meta: dict[str, dict] = {}

        for instance_id, entry in self.context_entries_snapshot().items():
            registered_gpu_ids.append(instance_id)
            ctx = entry.cache_context
            cache_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "kv_cache_layout": ctx.report_status(),
            }

        return {
            "registered_gpu_ids": registered_gpu_ids,
            "cache_context_meta": cache_context_meta,
        }

    def close(self) -> None:
        """Release GPU resources owned by this module."""
        # Stop the drain thread before storage_manager.close() so any
        # in-flight completions reach a live storage manager.
        self._device_host_func_dispatcher.stop()

        with self._lock:
            entries = list(self._cache_contexts.values())
            self._cache_contexts.clear()
        self._release_entries(entries)

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
            kv_caches: The KV cache tensor wrappers from the
                serving engine.
            model_name: The name of the model associated with this KV cache.
            world_size: The world size associated with this KV cache.
            engine_type: Which serving engine produced the caches.
                Forwarded to GPUCacheContext for format detection.
            layout_hints: See LayoutHints.  Forwarded to
                GPUCacheContext for GPU KV format detection.
            engine_group_infos: Engine-neutral KV cache group metadata
                (already msgspec-decoded by the message queue).
        """
        now = time.monotonic()
        # NOOP-register: an already-registered instance (e.g. a recovering
        # worker re-registering on its first ping) refreshes its last-seen
        # time so a stale entry is not reaped right after recovery. REGISTER
        # is SYNC-serialized on the MQ main loop, so it is the sole inserter.
        with self._lock:
            existing = self._cache_contexts.get(instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered; refreshing liveness",
                    instance_id,
                )
                return

        # Build the context and layout descriptor outside the lock.
        cache_context = create_cache_context(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            engine_group_infos=engine_group_infos,
            engine_type=engine_type,
            separate_object_groups=self._ctx.separate_object_groups,
            full_sw_kv=self._ctx.full_sw_kv,
        )
        kv_groups_manager = cache_context.kv_layer_groups_manager
        num_object_groups = kv_groups_manager.num_object_groups
        event_backend = get_event_ipc_backend(cache_context.device)
        event_backend.check_event_support(cache_context.device)
        layout_desc = get_layout_desc(
            cache_context, self._ctx.chunk_size, object_group_id=0
        )
        # One layout per object group, also in the single-group case: no
        # None special-casing downstream (group 0 maps to the merged layout).
        group_layout_descs = {
            gid: get_layout_desc(
                cache_context, self._ctx.chunk_size, object_group_id=gid
            )
            for gid in range(num_object_groups)
        }
        attn_desc = kv_groups_manager.get_attn_desc()
        self._ctx.layout_desc_registry.register(
            model_name,
            world_size,
            layout_desc,
            attn_desc,
            group_layout_descs=group_layout_descs,
        )

        with self._lock:
            self._cache_contexts[instance_id] = ContextEntry(
                cache_context=cache_context,
                model_name=model_name,
                world_size=world_size,
                last_seen=now,
                has_liveness_signal=False,
                event_backend=event_backend,
            )

        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            cache_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
        """
        with self._lock:
            popped = [
                e
                for e in (self._cache_contexts.pop(instance_id, None),)
                if e is not None
            ]
        if not popped:
            logger.warning(
                "No registered GPU context found for instance ID %d", instance_id
            )
            return

        # No scalar binding: `popped` must stay the only reference so
        # _release_entries' reclaim actually unmaps the IPC segments.
        self._release_entries(popped)
        logger.info("Unregistered KV cache for GPU ID %d", instance_id)

    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the GPU KV cache blocks to CPU.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker store operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to store, indexed by LMCache KV
                group index.
            event_ipc_handle: The IPC handle of the event to wait on.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the store operation, and the second
            element indicates whether the store operation completed without a
            fatal error (not whether every requested chunk was stored; see
            Notes). The event handle is empty when no device work was submitted.

        Raises:
            RuntimeError: If the backend does not support IPC event handles.

        Notes:
            All-or-nothing. If ``gpu_block_ids`` do not fully cover every chunk
            ``key`` resolves to for every LMCache group (e.g. a caller/protocol
            bug), or a copy fails, the whole store is skipped and nothing is
            committed (logged at WARNING); a subsequent retrieve simply misses
            and the engine recomputes. The boolean result reports whether the
            store completed without such a failure.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            # The worker can reconnect to a replacement server before its next
            # registration probe. No device work was submitted in that window,
            # so return an empty completion-event handle and a terminal False
            # response instead of leaving the MQ future unanswered. Echoing the
            # producer handle would make the originating process import its own
            # IPC event, which is invalid on HIP.
            logger.warning(
                "Rejecting STORE for unregistered GPU instance ID %d",
                instance_id,
            )
            return b"", False
        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # NOTE: different engine groups may have different block sizes, so
        # ``blocks_per_chunk[i]`` is the number of blocks in one chunk for
        # group ``i``.
        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: every LMCache group must have block IDs covering all
            # chunks. A short list (e.g. a caller/protocol bug) would otherwise
            # drive the transfer kernel to read out-of-bounds GPU memory, so skip
            # the whole store and commit nothing rather than caching a partial or
            # garbage entry. A later request can store it once the block IDs are
            # complete. Checked on the raw block ids, before cutting drops the
            # per-chunk blocks that sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.warning(
                    "STORE block ID underflow for request_id=%s: each group needs "
                    "num_chunks * blocks_per_chunk block IDs for %d chunks "
                    "(per-group blocks_per_chunk=%s); skipping the store.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            # Chunks whose block ids are all the null block (e.g. align-mode
            # Mamba chunks holding no real state) carry no valid KV and must not
            # be committed. Computed on the raw block ids before downsampling
            # mutates them.
            skipped_chunks = all_null_chunk_masks(
                gpu_block_ids,
                cache_context.kv_layer_groups_manager.object_groups,
                blocks_per_chunk,
                num_chunks,
            )

            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )

            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)

            # CPU-synchronous sentinel: a GPU store is about to be enqueued.
            # Must be published via publish() (not publish_on_stream) so the
            # drain thread sees it before MP_REQUEST_END can race MP_STORE_END.
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(cache_context.device)},
                )
            )

            # Worker 0 only: bindings depend on token content alone, so one
            # report covers every rank's keys. Published before finish_write
            # is enqueued so the token bindings precede the write-finished
            # events on the bus.
            if key.worker_id == 0 and self._ctx.event_bus.has_subscribers(
                EventType.MP_TOKENS
            ):
                self._publish_token_bindings(key, obj_keys_per_obj_group[0])

            transfer_key = next_transfer_key(key.request_id)
            self._ctx.event_bus.publish_on_stream(
                cache_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(cache_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                        "transfer_key": transfer_key,
                    },
                ),
            )

            reserved_dict: dict[ObjectKey, MemoryObj] = {}
            all_dict: dict[ObjectKey, MemoryObj] = {}
            total_bytes: int = 0
            store_succeeded = False
            try:
                for obj_group_id in range(num_object_groups):
                    obj_keys = obj_keys_per_obj_group[obj_group_id]
                    skip_mask = skipped_chunks[obj_group_id]
                    keys_to_reserve = [
                        k for i, k in enumerate(obj_keys) if not skip_mask[i]
                    ]
                    layout_desc = get_layout_desc(
                        cache_context,
                        self._ctx.chunk_size,
                        object_group_id=obj_group_id,
                    )
                    reserved_dict = self._ctx.storage_manager.reserve_write(
                        keys_to_reserve, layout_desc, "new"
                    )
                    all_dict.update(reserved_dict)
                    if reserved_dict:
                        total_bytes += next(
                            iter(reserved_dict.values())
                        ).get_size() * len(reserved_dict)

                    # Keys not in reserved_dict (all-null chunks skipped above, or
                    # skipped by the storage manager) become None entries; the
                    # helper skips them for D2H.
                    memory_objs: list[MemoryObj | None] = [
                        reserved_dict.get(obj_key) for obj_key in obj_keys
                    ]

                    # NOTE: batch_size must stay 1 for store.
                    transfer_kv_per_object_group(
                        cache_context,
                        block_ids_per_group_gpu,
                        memory_objs,
                        object_group_id=obj_group_id,
                        batch_size=1,
                        skip_first_n_tokens=0,
                        direction=lmcache_native.TransferDirection.D2H,
                        transfer_key=transfer_key,
                    )

                store_succeeded = True
            except Exception:
                logger.exception("Cannot store keys due to exception")
            finally:
                event_backend.record_event(event, cache_context.stream)
                # Fail closed: commit the reserved objects only when every chunk
                # copied successfully; otherwise the whole store is skipped.
                stored_count = len(all_dict) if store_succeeded else 0
                if stored_count:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_write",
                        list(all_dict.keys()),
                    )
                else:
                    total_bytes = 0
                num_tokens = num_chunks * self._ctx.chunk_size if stored_count else 0
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": stored_count,
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                            "transfer_key": transfer_key,
                        },
                    ),
                )

        ed = time.perf_counter()
        if stored_count:
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_chunks * self._ctx.chunk_size,
                ed - st,
            )
        return (
            event_backend.export_event(event, cache_context.device),
            store_succeeded,
        )

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Retrieve the CPU KV cache and put into GPU blocks.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker retrieve operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to retrieve into, indexed by LMCache
                KV group index.
            event_ipc_handle: The IPC handle of the event to wait on.
            skip_first_n_tokens: Number of tokens to skip writing at
                the start of the retrieve range. This avoids overwriting
                APC-shared GPU blocks that may be read concurrently by other
                requests.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the retrieve operation, and the
            second element indicates whether the key was successfully retrieved.
            The event handle is empty when no device work was submitted.

        Raises:
            RuntimeError: If the backend does not support IPC event handles.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            # See store(): there is no completion event because no device work
            # was submitted. The False result lets the caller recover or
            # recompute without importing its own producer event.
            logger.warning(
                "Rejecting RETRIEVE for unregistered GPU instance ID %d",
                instance_id,
            )
            try:
                self._release_failed_retrieve_locks(key, instance_id)
            except Exception:
                # A cleanup failure must never suppress the terminal response:
                # the client otherwise waits forever because blocking-handler
                # exceptions are only logged by the MQ server.
                logger.exception(
                    "Failed to release RETRIEVE locks for unregistered "
                    "GPU instance ID %d",
                    instance_id,
                )
            return b"", False
        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        # Must be published via publish() (not publish_on_stream) so the
        # drain thread sees it before MP_REQUEST_END can race MP_RETRIEVE_END.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(cache_context.device)},
            )
        )

        transfer_key = next_transfer_key(key.request_id)
        self._ctx.event_bus.publish_on_stream(
            cache_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(cache_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                    "transfer_key": transfer_key,
                },
            ),
        )

        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: a short block-id list would drive the transfer
            # kernel to write out-of-bounds GPU memory. Checked on the raw
            # block ids, before cutting drops the per-chunk blocks that
            # sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.error(
                    "RETRIEVE block ID underflow for request_id=%s: each group "
                    "needs num_chunks * blocks_per_chunk block IDs for %d "
                    "chunks (per-group blocks_per_chunk=%s); skipping the "
                    "retrieve.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            # Cut and stage all block_ids to GPU once before the transfer
            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )
            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)

            # Per object group, the prefetch only locked the in-window suffix
            # (the last ``num_chunks_in_sw`` chunks; the whole prefix for full
            # attention, where the value is < 0). Read and transfer only those.
            # Aux (connector-private) groups are never served by the
            # std retrieve: the lookup does not lock their keys and their
            # block-id entry is a placeholder -- reading them would be an
            # unlocked read of a plane nobody consumes here.
            attn_desc = cache_context.kv_layer_groups_manager.get_attn_desc()
            skipped_groups = {
                g for g, kind in enumerate(attn_desc.group_kinds) if kind == "aux"
            }
            group_skips = [
                0 if window < 0 else max(0, num_chunks - window)
                for window in attn_desc.num_chunks_in_sw
            ]
            expected_retained = sum(
                num_chunks - skip
                for g, skip in enumerate(group_skips)
                if g not in skipped_groups
            )

            prefetched_keys: list[ObjectKey] = []
            total_bytes = 0
            retrieve_succeeded = True
            try:
                for obj_group_id in range(num_object_groups):
                    if obj_group_id in skipped_groups:
                        continue
                    skip = group_skips[obj_group_id]
                    in_window_keys = obj_keys_per_obj_group[obj_group_id][skip:]
                    with self._ctx.storage_manager.read_prefetched_results(
                        in_window_keys
                    ) as window_objs:
                        if not window_objs or len(window_objs) != len(in_window_keys):
                            logger.error("Some keys not found during retrieve!")
                            retrieve_succeeded = False
                            break

                        total_bytes += sum(mo.get_size() for mo in window_objs)

                        # None-pad the skipped prefix to full length so the
                        # transfer's ``num_objects_to_skip`` and block-id slicing
                        # line up unchanged; the None entries are never read.
                        memory_objs: list[MemoryObj | None] = [None] * skip + list(
                            window_objs
                        )

                        transfer_kv_per_object_group(
                            cache_context,
                            block_ids_per_group_gpu,
                            memory_objs,
                            object_group_id=obj_group_id,
                            batch_size=cache_context.max_batch_size,
                            skip_first_n_tokens=skip_first_n_tokens,
                            direction=lmcache_native.TransferDirection.H2D,
                            transfer_key=transfer_key,
                        )
                        # Extend only after the copy is enqueued: on exception,
                        # read_prefetched_results releases this group's locks
                        # itself, and a key must not be released twice.
                        prefetched_keys.extend(in_window_keys)
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                retrieve_succeeded = False
            finally:
                event_backend.record_event(event, cache_context.stream)
                if prefetched_keys:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_read_prefetched",
                        prefetched_keys,
                    )
                num_tokens = (
                    num_chunks * self._ctx.chunk_size
                    if len(prefetched_keys) == expected_retained
                    else 0
                )
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                            "transfer_key": transfer_key,
                        },
                    ),
                )
        if retrieve_succeeded:
            tokens_retrieved = num_chunks * self._ctx.chunk_size
            ed = time.perf_counter()
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                tokens_retrieved,
                ed - st,
            )

        return (
            event_backend.export_event(event, cache_context.device),
            retrieve_succeeded,
        )

    @_lmcache_nvtx_annotate
    def retrieve_layerwise(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[list[bytes], bool]:
        """Retrieve KV cache blocks into GPU, returning one event per layer.

        Same payload, lock discipline, and failure semantics as
        :meth:`retrieve`, but the server splits the H2D transfer per local
        (PP-stage) layer and records one CUDA event per layer, in local layer
        order. The first element of the response is therefore a list of event
        IPC handles -- one per layer of this instance's KV cache.

        When per-layer transfer is not applicable for this deployment (GDS
        objects, multi-kernel-group / sliding-window object groups, MLA /
        cross-layer / SGLang 2-plane KV formats), the server falls back to the
        whole-object transfer and duplicates the single completion event once
        per layer, so the worker's per-layer waits remain correct (they all
        resolve when the whole transfer completes, just without overlap).

        Returns:
            A tuple of (list of event handles, success flag). The event handle
            list has exactly ``cache_context.num_layers`` entries (the local
            layer count of this PP stage); the flag indicates whether the key
            was successfully retrieved.

        Raises:
            RuntimeError: If the backend does not support IPC event handles.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            # Same as retrieve(): no completion events because no device work
            # was submitted; the empty list lets the caller fail closed.
            logger.warning(
                "Rejecting RETRIEVE_LAYERWISE for unregistered GPU instance ID %d",
                instance_id,
            )
            try:
                self._release_failed_retrieve_locks(key, instance_id)
            except Exception:
                logger.exception(
                    "Failed to release RETRIEVE locks for unregistered "
                    "GPU instance ID %d",
                    instance_id,
                )
            return [], False
        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])
        # Local (PP-stage) layer count: per-layer events are generated for
        # exactly the layers this instance owns (e.g. 78 layers over PP4 ->
        # 22/20/20/16 per stage).
        num_layers = cache_context.num_layers

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(cache_context.device)},
            )
        )

        transfer_key = next_transfer_key(key.request_id)
        self._ctx.event_bus.publish_on_stream(
            cache_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(cache_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                    "transfer_key": transfer_key,
                },
            ),
        )

        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: a short block-id list would drive the transfer
            # kernel to write out-of-bounds GPU memory (see retrieve()).
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.error(
                    "RETRIEVE_LAYERWISE block ID underflow for request_id=%s: "
                    "each group needs num_chunks * blocks_per_chunk block IDs "
                    "for %d chunks (per-group blocks_per_chunk=%s); skipping "
                    "the retrieve.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                handle = event_backend.export_event(event, cache_context.device)
                return [handle] * max(1, num_layers), False

            # Cut and stage all block_ids to GPU once before the transfer
            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )
            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)

            attn_desc = cache_context.kv_layer_groups_manager.get_attn_desc()
            skipped_groups = {
                g for g, kind in enumerate(attn_desc.group_kinds) if kind == "aux"
            }
            group_skips = [
                0 if window < 0 else max(0, num_chunks - window)
                for window in attn_desc.num_chunks_in_sw
            ]
            expected_retained = sum(
                num_chunks - skip
                for g, skip in enumerate(group_skips)
                if g not in skipped_groups
            )

            prefetched_keys: list[ObjectKey] = []
            total_bytes = 0
            retrieve_succeeded = True
            per_layer_events: list[Any] = []

            # Structural eligibility is uniform per cache context; GDS objects
            # are uniform per L1 backend and are probed on the first read.
            can_layerwise = (
                num_object_groups == 1
                and 0 not in skipped_groups
                and num_layers > 0
                and _object_group_layerwise_supported(cache_context, 0)
            )

            try:
                if can_layerwise:
                    # Single object group: hold its read locks across every
                    # layer so the chunks stay resident for the whole loop.
                    obj_group_id = 0
                    skip = group_skips[obj_group_id]
                    in_window_keys = obj_keys_per_obj_group[obj_group_id][skip:]
                    with self._ctx.storage_manager.read_prefetched_results(
                        in_window_keys
                    ) as window_objs:
                        if not window_objs or len(window_objs) != len(in_window_keys):
                            logger.error("Some keys not found during retrieve!")
                            retrieve_succeeded = False
                        else:
                            if any(
                                isinstance(mo, GDSMemoryObject)
                                for mo in window_objs
                            ):
                                # GDS reads are whole-object; the per-layer path
                                # cannot slice them. Fall through to the
                                # whole-object transfer below (re-reads the
                                # objects with fresh locks).
                                can_layerwise = False
                            else:
                                memory_objs: list[MemoryObj | None] = [None] * skip
                                memory_objs += list(window_objs)
                                total_bytes += sum(
                                    mo.get_size() for mo in window_objs
                                )
                                for layer_idx in range(num_layers):
                                    transfer_kv_per_object_group_layerwise(
                                        cache_context,
                                        block_ids_per_group_gpu,
                                        memory_objs,
                                        object_group_id=obj_group_id,
                                        batch_size=cache_context.max_batch_size,
                                        skip_first_n_tokens=skip_first_n_tokens,
                                        layer_idx=layer_idx,
                                        num_layers=num_layers,
                                        transfer_key=transfer_key,
                                    )
                                    layer_event = event_backend.create_event(
                                        cache_context.device
                                    )
                                    event_backend.record_event(
                                        layer_event, cache_context.stream
                                    )
                                    per_layer_events.append(layer_event)
                                # Extend only after all copies are enqueued.
                                prefetched_keys.extend(in_window_keys)

                if not can_layerwise:
                    # Whole-object fallback: identical loop to retrieve(), but
                    # the single completion event is duplicated per layer.
                    for obj_group_id in range(num_object_groups):
                        if obj_group_id in skipped_groups:
                            continue
                        skip = group_skips[obj_group_id]
                        in_window_keys = obj_keys_per_obj_group[obj_group_id][skip:]
                        with self._ctx.storage_manager.read_prefetched_results(
                            in_window_keys
                        ) as window_objs:
                            if not window_objs or len(window_objs) != len(
                                in_window_keys
                            ):
                                logger.error("Some keys not found during retrieve!")
                                retrieve_succeeded = False
                                break

                            total_bytes += sum(mo.get_size() for mo in window_objs)

                            memory_objs = [None] * skip + list(window_objs)

                            transfer_kv_per_object_group(
                                cache_context,
                                block_ids_per_group_gpu,
                                memory_objs,
                                object_group_id=obj_group_id,
                                batch_size=cache_context.max_batch_size,
                                skip_first_n_tokens=skip_first_n_tokens,
                                direction=lmcache_native.TransferDirection.H2D,
                                transfer_key=transfer_key,
                            )
                            prefetched_keys.extend(in_window_keys)
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                retrieve_succeeded = False
            finally:
                if not per_layer_events:
                    # No per-layer events recorded (fallback or failure): the
                    # single event covers the whole transfer.
                    event_backend.record_event(event, cache_context.stream)
                if prefetched_keys:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_read_prefetched",
                        prefetched_keys,
                    )
                num_tokens = (
                    num_chunks * self._ctx.chunk_size
                    if len(prefetched_keys) == expected_retained
                    else 0
                )
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                            "transfer_key": transfer_key,
                        },
                    ),
                )
        if retrieve_succeeded:
            tokens_retrieved = num_chunks * self._ctx.chunk_size
            ed = time.perf_counter()
            logger.info(
                "Retrieved %d tokens in %.3f seconds (layerwise, %d layers)",
                tokens_retrieved,
                ed - st,
                num_layers,
            )

        if per_layer_events:
            handles = [
                event_backend.export_event(evt, cache_context.device)
                for evt in per_layer_events
            ]
        else:
            handle = event_backend.export_event(event, cache_context.device)
            handles = [handle] * max(1, num_layers)
        return handles, retrieve_succeeded

    def _publish_token_bindings(
        self, key: IPCCacheServerKey, obj_keys: list[ObjectKey]
    ) -> None:
        """Publish one ``MP_TOKENS`` event for ``key``'s chunks.

        Pairs each complete chunk in ``[key.start, key.end)`` with its
        ObjectKey chunk hash and token position. Must be called at store
        submission, before the write-finished events reach the bus, so the
        cache-event subscriber can stamp them onto the STORE entries. A
        store that later fails leaves only unused cache entries.

        Args:
            key: The IPC key of the store being submitted.
            obj_keys: One ObjectKey per complete chunk, in chunk order.
        """
        # Complete chunks in [key.start, key.end) paired with the absolute
        # position of each chunk's first token. Prefix-chained chunk hashes
        # imply a position without revealing it, so it is reported here. A
        # trailing partial chunk has no stored KV to bind to.
        chunk_size = self._ctx.chunk_size
        token_ids = list(key.token_ids)
        effective_len = min(len(token_ids), key.end)
        num_complete = effective_len - effective_len % chunk_size
        token_offsets = list(range(key.start, num_complete, chunk_size))
        token_chunks = [
            token_ids[offset : offset + chunk_size] for offset in token_offsets
        ]
        if not token_chunks:
            return
        if len(obj_keys) != len(token_chunks):
            logger.warning(
                "Skipping token bindings for request %s: %d resolved keys "
                "vs %d complete chunks in [%d, %d)",
                key.request_id,
                len(obj_keys),
                len(token_chunks),
                key.start,
                key.end,
            )
            return
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_TOKENS,
                session_id=key.request_id,
                metadata={
                    "chunk_hashes": [obj_key.chunk_hash for obj_key in obj_keys],
                    "token_chunks": token_chunks,
                    "token_offsets": token_offsets,
                },
            )
        )
