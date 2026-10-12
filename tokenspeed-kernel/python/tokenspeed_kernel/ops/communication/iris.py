# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Prepared Iris collectives sharing one context and stable symmetric storage.

The runtime owns one all-reduce state per group and dtype. Each protocol owns
its epochs and scratch; model-specific fusions borrow those buffers explicitly.
"""

import logging
from functools import cache
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import gl, triton
from tokenspeed_kernel.platform import current_platform

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication._iris import IrisGroup

logger = logging.getLogger(__file__)
_platform = current_platform()
IRIS_ALL_REDUCE_MAX_BYTES = 512 << 10
_PACKED_WORD_BYTES = 8
_TWO_SHOT_WORLD_SIZES = (4, 8)
_PRODUCER_TWO_SHOT_MIN_BYTES = {4: 160 << 10, 8: 96 << 10}
_PRODUCER_DIRECT_GL_DTYPES = {
    torch.bfloat16: gl.bfloat16,
    torch.float16: gl.float16,
    torch.float32: gl.float32,
}
IRIS_AR_RMSNORM_STATES: dict = {}


def configure_iris(*, heap_size: int | None, allocator_type: str | None) -> None:
    """Configure the shared context before preparing any Iris collective.

    Args:
        heap_size: Initial heap capacity in bytes, or None for automatic sizing.
        allocator_type: Iris allocator name, or None for Iris's default. Iris's
            ``IRIS_ALLOCATOR`` environment override retains its normal precedence.
    """
    from tokenspeed_kernel.ops.communication._iris import configure_iris as configure

    configure(heap_size=heap_size, allocator_type=allocator_type)


def producer_direct_all_reduce_can_run(
    world_size: int,
    total_numel: int,
    dtype: torch.dtype,
    max_bytes: int,
) -> bool:
    """Check the generic AMD producer-direct kernel's payload requirements.

    Args:
        world_size: Number of ranks participating in the all-reduce.
        total_numel: Total number of elements in the payload.
        dtype: Element type of the payload.
        max_bytes: Byte capacity of the producer-direct symmetric input buffer.

    Returns:
        Whether the group size and element type are supported and the payload is
        positive, packed-word aligned, and within the input-buffer capacity.
    """
    element_bytes = dtype.itemsize
    return (
        world_size in (2, 4, 8)
        and total_numel > 0
        and dtype in _PRODUCER_DIRECT_GL_DTYPES
        and _PACKED_WORD_BYTES % element_bytes == 0
        and total_numel % (_PACKED_WORD_BYTES // element_bytes) == 0
        and total_numel * element_bytes <= max_bytes
    )


def _use_two_stage_producer_direct(
    world_size: int,
    total_numel: int,
    dtype: torch.dtype,
) -> bool:
    min_bytes = _PRODUCER_TWO_SHOT_MIN_BYTES.get(world_size)
    return (
        min_bytes is not None
        and total_numel * dtype.itemsize >= min_bytes
        and _use_two_stage_plain(world_size, total_numel, dtype)
    )


def _use_two_stage_plain(
    world_size: int,
    numel: int,
    dtype: torch.dtype,
) -> bool:
    """Use two-shot when the input splits into whole packed words per rank."""
    return (
        dtype in _PRODUCER_DIRECT_GL_DTYPES
        and world_size in _TWO_SHOT_WORLD_SIZES
        and numel % (world_size * (_PACKED_WORD_BYTES // dtype.itemsize)) == 0
    )


def _get_available_gpu_memory(gpu_id: int, empty_cache: bool = True) -> float:
    if torch.cuda.is_available():
        with torch.cuda.device(gpu_id):
            if empty_cache:
                torch.cuda.empty_cache()
            free_gpu_memory, _ = torch.cuda.mem_get_info()
            return free_gpu_memory / (1 << 30)
    return 0.0


class IrisAllReduce:
    """Own one group's prepared staged and producer-direct all-reduce storage."""

    @staticmethod
    def heap_bytes(
        staged_max_numel: int,
        producer_direct_max_numel: int,
        world_size: int,
        dtype: torch.dtype,
    ) -> int:
        """Return symmetric storage bytes for the group's element capacities."""
        from tokenspeed_kernel.ops.communication._iris.all_reduce import (
            all_reduce_heap_bytes,
        )

        return all_reduce_heap_bytes(
            staged_max_numel, producer_direct_max_numel, world_size, dtype
        )

    def __init__(
        self,
        peers: "IrisGroup",
        staged_max_numel: int,
        producer_direct_max_numel: int,
        dtype: torch.dtype,
    ):
        from tokenspeed_kernel.ops.communication._iris.all_reduce import (
            ProducerBuffer,
            ProducerPull,
            StagedPullOneShot,
            StagedPullTwoShot,
            staged_tuning,
        )

        if min(staged_max_numel, producer_direct_max_numel) < 0:
            raise ValueError("Iris all-reduce capacities must be non-negative")
        self.peers = peers
        self.group = peers.group
        self.rank_in_group = peers.rank
        self.world_size = peers.world_size
        self.device = peers.device
        self.dtype = dtype
        self.staged_max_numel = staged_max_numel
        self.producer_direct_max_numel = producer_direct_max_numel
        self.producer = (
            ProducerBuffer(peers, producer_direct_max_numel, dtype)
            if producer_direct_max_numel
            else None
        )
        self.producer_pull = (
            ProducerPull(self.producer) if self.producer is not None else None
        )
        self.staged = (
            StagedPullOneShot(
                peers,
                staged_max_numel,
                dtype,
                block_size=StagedPullOneShot.BLOCK_SIZE,
                num_subgroups=StagedPullOneShot.NUM_SUBGROUPS,
            )
            if staged_max_numel
            else None
        )
        tuning = staged_tuning(staged_max_numel, peers.world_size, dtype)
        self.tuned_staged = (
            StagedPullOneShot(
                peers, tuning[0], dtype, block_size=tuning[1], num_subgroups=tuning[2]
            )
            if tuning is not None
            else None
        )
        self.staged_two_shot = (
            StagedPullTwoShot(peers, staged_max_numel, dtype)
            if staged_max_numel and StagedPullTwoShot.supported(peers.world_size, dtype)
            else None
        )

    def all_reduce(
        self, tensor: torch.Tensor, op=None, safe: bool = True, async_op: bool = False
    ) -> torch.Tensor:
        """Reduce in place; optionally return a separate copy of the result."""
        if op is None:
            op = dist.ReduceOp.SUM
        assert op == dist.ReduceOp.SUM, f"Iris all-reduce only supports SUM, got {op}"
        assert not async_op, "Iris all-reduce does not support async_op"
        assert tensor.dtype == self.dtype
        assert 0 < tensor.numel() <= self.staged_max_numel
        if (
            self.tuned_staged is not None
            and tensor.numel() == self.tuned_staged.slots.shape[1]
        ):
            self.tuned_staged.run(tensor, tensor)
        elif self.staged_two_shot is not None and self.staged_two_shot.can_run(
            tensor.numel()
        ):
            self.staged_two_shot.run(tensor, tensor)
        else:
            assert self.staged is not None
            self.staged.run(tensor, tensor)
        return tensor.clone() if safe else tensor

    def acquire_outputs(
        self, shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[torch.Tensor, ...]:
        """Borrow consecutive symmetric views until the next ordered producer use."""
        assert self.producer is not None
        return self.producer.acquire(shapes)

    def owns_outputs(self, tensors: tuple[torch.Tensor, ...]) -> bool:
        return self.producer is not None and self.producer.owns(tensors)

    def all_reduce_symmetric(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Reduce producer outputs into caller-owned local storage."""
        assert self.owns_outputs(tensors)
        assert self.producer_pull is not None
        shapes = tuple(tuple(tensor.shape) for tensor in tensors)
        if not self.producer_pull.can_run(shapes):
            raise RuntimeError(
                "Prepared Iris producer reduction does not support these outputs"
            )
        output = torch.empty(
            sum(tensor.numel() for tensor in tensors),
            dtype=self.dtype,
            device=self.device,
        )
        self.producer_pull.run(output)
        return self.producer.views(output, shapes)


def create_iris_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    staged_max_numel: int,
    producer_direct_max_numel: int,
    dtype: torch.dtype,
    heap_size: int | None,
    device: torch.device | None,
) -> IrisAllReduce:
    """Prepare generic staged and producer pull implementations.

    Args:
        group: Collective process group.
        rank_in_group: This process's rank within the group.
        staged_max_numel: Maximum payload staged from ordinary tensors.
        producer_direct_max_numel: Maximum payload written directly by producers.
        dtype: Element type of the payload.
        heap_size: Initial shared heap capacity in bytes, or None to size it.
        device: Owning device, or None for the current device.

    Returns:
        A prepared operation. Its implementations own storage and synchronization.
    """
    assert type(group) == dist.ProcessGroup
    assert dist.is_initialized() and _platform.is_amd
    if min(staged_max_numel, producer_direct_max_numel) < 0:
        raise ValueError("Iris all-reduce capacities must be non-negative")
    from tokenspeed_kernel.ops.communication._iris import IrisGroup, reserve_heap

    device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
    ctx = reserve_heap(
        IrisAllReduce.heap_bytes(
            staged_max_numel, producer_direct_max_numel, group.size(), dtype
        ),
        heap_size=heap_size,
    )
    peers = IrisGroup(ctx, group, rank_in_group, device)
    state = IrisAllReduce(peers, staged_max_numel, producer_direct_max_numel, dtype)
    peers.finish_prepare()
    return state


def iris_all_reduce(
    state: "IrisAllReduce",
    tensor: torch.Tensor,
    op=None,
    safe: bool = True,
    async_op: bool = False,
) -> torch.Tensor:
    return state.all_reduce(tensor, op=op, safe=safe, async_op=async_op)


def iris_acquire_outputs(
    state: "IrisAllReduce",
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, ...]:
    """Return consecutive symmetric producer-output views for Iris."""
    return state.acquire_outputs(shapes)


def iris_all_reduce_symmetric(
    state: "IrisAllReduce",
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Return caller-owned reductions of consecutive symmetric producer outputs."""
    return state.all_reduce_symmetric(tensors)


@cache
def _iris_ccl():
    import importlib
    import pkgutil

    from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

    with redirect_triton_to_tokenspeed_triton():
        import iris.ccl.triton
        from iris.ccl import Config
        from iris.ccl.all_gather import all_gather
        from iris.ccl.reduce_scatter import reduce_scatter

        for info in pkgutil.walk_packages(
            iris.ccl.triton.__path__, prefix="iris.ccl.triton."
        ):
            importlib.import_module(info.name)
    return Config, all_gather, reduce_scatter


class IrisRSAG(object):

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_tokens: int,
        hidden_size: int,
        device: torch.device = None,
        heap_size: int | None = None,
    ) -> None:
        assert (
            type(group) == dist.ProcessGroup
        ), f"Expected dist.ProcessGroup, got {type(group)}"
        assert dist.is_initialized(), (
            "torch.distributed must be initialized before constructing "
            "IrisRSAG; call dist.init_process_group() first."
        )
        assert _platform.is_amd, (
            "IrisRSAG currently targets AMD ROCm; " f"got non-AMD platform: {_platform}"
        )
        assert (
            group == dist.group.WORLD or group.size() == dist.get_world_size()
        ), "iris.ccl all_gather/reduce_scatter do not accept a sub-group."

        self.group = group
        self.rank_in_group = rank_in_group
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.dtype = torch.bfloat16
        self.world_size = group.size()

        # Heap holds in/out flat buffers plus iris bookkeeping; over-provision
        # similarly to ``IrisAllReduce`` to leave room for ring/spinlock flags.
        if heap_size is None:
            buf_bytes = max_tokens * hidden_size * self.dtype.itemsize
            heap_size = max(1 << 28, 4 * buf_bytes + (16 << 20))

        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        from tokenspeed_kernel.ops.communication._iris import (
            allocation_bytes,
            reserve_heap,
        )

        self._ctx = reserve_heap(
            2 * allocation_bytes((max_tokens, hidden_size), self.dtype),
            heap_size=heap_size,
        )
        self._in_buff = self._ctx.empty((max_tokens, hidden_size), dtype=self.dtype)
        self._out_buff = self._ctx.empty((max_tokens, hidden_size), dtype=self.dtype)
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris RSAG symmetric-heap buffers allocated: "
            f"{free_gpu_memory_begin - free_gpu_memory_after!s} GB",
        )

        assert self._ctx.get_num_ranks() == dist.get_world_size(), (
            f"Iris world size {self._ctx.get_num_ranks()} "
            f"!= torch world size {dist.get_world_size()}"
        )
        assert self.rank_in_group == self._ctx.get_rank(), (
            f"rank mismatch: rank_in_group={self.rank_in_group}, "
            f"iris rank={self._ctx.get_rank()}"
        )

    # -- token-distribution helpers (mirror sibling classes) ----------------

    def get_token_dist(self, total_tokens_in_group: int) -> list:
        token_list_in_group = []
        for rank in range(self.world_size):
            num_tokens_per_rank = total_tokens_in_group // self.world_size + (
                1 if (rank < total_tokens_in_group % self.world_size) else 0
            )
            token_list_in_group.append(num_tokens_per_rank)
        return token_list_in_group

    def get_context(self, token_list_in_group: list) -> Tuple[int, int, int]:
        total_num_tokens = sum(token_list_in_group)
        assert (
            total_num_tokens <= self.max_tokens
        ), f"The inner comm buffer is too small: {total_num_tokens=} is not <= {self.max_tokens=}"
        local_num_tokens = token_list_in_group[self.rank_in_group]
        local_token_offset = sum(token_list_in_group[: self.rank_in_group])
        return total_num_tokens, local_num_tokens, local_token_offset

    # -- internal helpers ---------------------------------------------------

    def _assert_uniform(self, token_list_in_group: List[int]) -> int:
        first = token_list_in_group[0]
        assert all(t == first for t in token_list_in_group), (
            "IrisRSAG requires uniform tokens per rank; got "
            f"token_list_in_group={token_list_in_group}"
        )
        return first

    @staticmethod
    def _pick_block_n(hidden_size: int) -> int:
        # Pick the largest power-of-two block that divides hidden_size, capped
        # at 256. This keeps the iris kernel on its no-mask fast path and
        # still produces enough tiles (world_size * hidden/block_n) to fill
        # ``comm_sms`` SMs on supported AMD chips.
        for cand in (256, 128, 64, 32, 16):
            if hidden_size % cand == 0:
                return cand
        return hidden_size

    def _make_config(self, local_num_tokens: int, hidden_size: int):
        # ``swizzle_size=1`` keeps tile_id ordering row-major in M, which is
        # required so that block-distribution (DISTRIBUTION=1) hands rank r
        # exactly the K tiles spanning rows [r*local, (r+1)*local) in the
        # reduce-scatter kernel. ``all_gather`` is rank-agnostic on tile order
        # so the same config is fine.
        config_type, _, _ = _iris_ccl()
        return config_type(
            block_size_m=local_num_tokens,
            block_size_n=self._pick_block_n(hidden_size),
            swizzle_size=1,
            all_reduce_distribution=1,
        )

    # -- public collective ops ---------------------------------------------

    def reduce_scatter(
        self,
        hidden_states: torch.Tensor,
        tp_num_tokens: int = None,
        token_list_in_group: List[int] = None,
        safe=True,
    ) -> torch.Tensor:
        assert (
            tp_num_tokens is not None or token_list_in_group is not None
        ), "Either tp_num_tokens or token_list_in_group must be provided"
        if token_list_in_group is None:
            token_list_in_group = self.get_token_dist(tp_num_tokens)
        assert (
            hidden_states.dtype == self.dtype
        ), f"Only {self.dtype} is supported, got {hidden_states.dtype}"

        local_num_tokens = self._assert_uniform(token_list_in_group)
        total_num_tokens, _, local_token_offset = self.get_context(token_list_in_group)
        assert (hidden_states.shape[0] == total_num_tokens) and (
            hidden_states.shape[-1] == self.hidden_size
        ), (
            f"Mismatched shape, {hidden_states.shape[0]=} != {total_num_tokens=} "
            f"or {hidden_states.shape[-1]=} != {self.hidden_size=} "
            f"{hidden_states.shape=}"
        )

        if local_num_tokens == 0:
            return torch.empty(
                (0, self.hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        in_view = self._in_buff[:total_num_tokens, : self.hidden_size]
        out_view = self._out_buff[:total_num_tokens, : self.hidden_size]
        in_view.copy_(hidden_states)

        # Ensure every rank's shared input copy is visible before peer loads begin.
        self._ctx.device_barrier()

        config = self._make_config(local_num_tokens, self.hidden_size)
        _, _, reduce_scatter = _iris_ccl()
        reduce_scatter(out_view, in_view, self._ctx, config=config)

        output = out_view[local_token_offset : local_token_offset + local_num_tokens, :]
        return output.clone() if safe else output

    def all_gather(
        self,
        hidden_states: torch.Tensor,
        tp_num_tokens: int = None,
        token_list_in_group: List[int] = None,
        safe=True,
    ) -> torch.Tensor:
        assert (
            tp_num_tokens is not None or token_list_in_group is not None
        ), "Either tp_num_tokens or token_list_in_group must be provided"
        if token_list_in_group is None:
            token_list_in_group = self.get_token_dist(tp_num_tokens)
        assert (
            hidden_states.dtype == self.dtype
        ), f"Only {self.dtype} is supported, got {hidden_states.dtype}"

        local_num_tokens = self._assert_uniform(token_list_in_group)
        total_num_tokens, _, _ = self.get_context(token_list_in_group)
        hidden_size = hidden_states.shape[-1]
        assert (hidden_states.shape[0] == local_num_tokens) and (
            hidden_size <= self.hidden_size
        ), (
            f"{hidden_states.shape=}|{local_num_tokens=}|{hidden_states.device=} "
            "Mismatched shape"
        )

        if local_num_tokens == 0:
            return torch.empty(
                (0, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        in_view = self._in_buff[:local_num_tokens, :hidden_size]
        out_view = self._out_buff[:total_num_tokens, :hidden_size]
        in_view.copy_(hidden_states)

        self._ctx.device_barrier()

        config = self._make_config(local_num_tokens, hidden_size)
        _, all_gather, _ = _iris_ccl()
        all_gather(out_view, in_view, self._ctx, config=config)

        return out_view.clone() if safe else out_view


class IrisAllReduceResidualRMSNorm(object):

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        heap_size: int | None = None,
        device: torch.device = None,
        persistent: bool = False,
    ) -> None:
        assert (
            type(group) == dist.ProcessGroup
        ), f"Expected dist.ProcessGroup, got {type(group)}"
        assert dist.is_initialized(), (
            "torch.distributed must be initialized before constructing "
            "IrisAllReduceResidualRMSNorm; call dist.init_process_group() first."
        )
        assert _platform.is_amd, (
            "IrisAllReduceResidualRMSNorm currently targets AMD ROCm; "
            f"got non-AMD platform: {_platform}"
        )

        self.group = group
        self.rank_in_group = rank_in_group
        self.world_size = group.size()
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")

        if heap_size is None:
            buf_bytes = max_token_num * hidden_dim * dtype.itemsize
            heap_size = max(1 << 28, 4 * buf_bytes + (16 << 20))
        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        from tokenspeed_kernel.ops.communication._iris import (
            allocation_bytes,
            reserve_heap,
        )

        self._ctx = reserve_heap(
            allocation_bytes((max_token_num, hidden_dim), dtype), heap_size=heap_size
        )
        self._input_buf = self._ctx.zeros((max_token_num, hidden_dim), dtype=dtype)
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris AR+RMSNorm symmetric-heap buffer allocated: "
            f"{free_gpu_memory_begin - free_gpu_memory_after!s} GB",
        )

        self._rank_start = 0
        self._rank_stride = 1
        self._iris_rank = dist.get_rank()

        self.persistent = persistent
        self._num_programs = (
            torch.cuda.get_device_properties(self.device).multi_processor_count
            if persistent
            else 0
        )

    def fused(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        norm_out: torch.Tensor | None = None,
        residual_out: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert input_tensor.dtype == self.dtype, (
            f"Iris AR+RMSNorm dtype mismatch: input={input_tensor.dtype}, "
            f"backend={self.dtype}"
        )
        assert input_tensor.dim() == 2, (
            f"input must be 2-D (num_tokens, hidden_dim), got "
            f"shape={input_tensor.shape}"
        )
        assert (
            input_tensor.shape == residual.shape
        ), f"residual shape {residual.shape} != input shape {input_tensor.shape}"
        assert input_tensor.shape[1] == self.hidden_dim, (
            f"hidden_dim mismatch: input={input_tensor.shape[1]} vs "
            f"backend={self.hidden_dim}"
        )
        num_tokens = input_tensor.shape[0]
        assert num_tokens <= self.max_token_num, (
            f"num_tokens ({num_tokens}) exceeds max_token_num "
            f"({self.max_token_num})"
        )
        assert weight.shape == (
            self.hidden_dim,
        ), f"weight shape {weight.shape} != ({self.hidden_dim},)"
        assert input_tensor.is_contiguous() and residual.is_contiguous()

        from tokenspeed_kernel.ops.communication._iris.all_reduce import (
            iris_pull_oneshot_rmsnorm,
            iris_pull_oneshot_rmsnorm_persistent,
        )

        in_view = self._input_buf[:num_tokens, :]
        in_view.copy_(input_tensor)

        if norm_out is None:
            norm_out = torch.empty_like(input_tensor)
        if residual_out is None:
            residual_out = torch.empty_like(residual)

        self._ctx.device_barrier()

        heap_bases = self._ctx.get_heap_bases()
        BLOCK_SIZE = triton.next_power_of_2(self.hidden_dim)
        if self.persistent:
            kernel = iris_pull_oneshot_rmsnorm_persistent
            grid = (min(num_tokens, self._num_programs),)
        else:
            kernel = iris_pull_oneshot_rmsnorm
            grid = (num_tokens,)
        kernel[grid](
            in_view,
            residual,
            weight,
            norm_out,
            residual_out,
            num_tokens,
            heap_bases,
            iris_rank=self._iris_rank,
            world_size=self.world_size,
            rank_start=self._rank_start,
            rank_stride=self._rank_stride,
            HIDDEN_SIZE=self.hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
            EPS=eps,
            num_warps=8,
        )
        # Ensure all peer loads finish before the next call reuses _input_buf.
        self._ctx.device_barrier()
        return norm_out, residual_out


def create_iris_rsag_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_tokens: int,
    hidden_size: int,
    device: torch.device = None,
    heap_size: int | None = None,
) -> "IrisRSAG":
    return IrisRSAG(
        group=group,
        rank_in_group=rank_in_group,
        max_tokens=max_tokens,
        hidden_size=hidden_size,
        device=device,
        heap_size=heap_size,
    )


def create_iris_ar_rmsnorm_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    heap_size: int | None = None,
    device: torch.device = None,
    persistent: bool = False,
) -> "IrisAllReduceResidualRMSNorm":
    return IrisAllReduceResidualRMSNorm(
        group=group,
        rank_in_group=rank_in_group,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        heap_size=heap_size,
        device=device,
        persistent=persistent,
    )


def iris_allreduce_residual_rmsnorm(
    state: "IrisAllReduceResidualRMSNorm",
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    norm_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return state.fused(
        input_tensor=input_tensor,
        residual=residual,
        weight=weight,
        eps=eps,
        norm_out=norm_out,
        residual_out=residual_out,
    )
