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

"""One process-wide Iris heap; operations retain their own group and buffers."""

import math

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

with redirect_triton_to_tokenspeed_triton():
    import iris

MIN_HEAP_BYTES = 256 << 20
HEAP_HEADROOM_BYTES = 16 << 20
_context = None
_reserved_bytes = 0
_heap_size: int | None = None
_allocator_type: str | None = None


def configure_iris(*, heap_size: int | None, allocator_type: str | None) -> None:
    """Set heap options before preparation; None retains Iris's allocator default.

    The allocator name is passed unchanged to Iris. Its IRIS_ALLOCATOR environment
    override retains upstream precedence. Neither option may change a live heap.
    """
    global _heap_size, _allocator_type
    if _context is not None and (heap_size, allocator_type) != (
        _heap_size,
        _allocator_type,
    ):
        raise RuntimeError("Configure Iris before preparing communication buffers")
    _heap_size = heap_size
    _allocator_type = allocator_type


def allocation_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    """Account for Iris's 1024-byte symmetric suballocation alignment."""
    return (math.prod(shape) * dtype.itemsize + 1023) // 1024 * 1024


def reserve_heap(num_bytes: int, *, heap_size: int | None):
    """Reserve new operations' combined storage before any graph captures it."""
    global _context, _reserved_bytes
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare Iris communication buffers before graph capture")
    required = _reserved_bytes + num_bytes
    if _context is None:
        size = _heap_size if _heap_size is not None else heap_size
        if size is None:
            size = max(MIN_HEAP_BYTES, required + HEAP_HEADROOM_BYTES)
        if size < required:
            raise ValueError(
                "Iris heap is smaller than the prepared communication buffers"
            )
        options = {} if _allocator_type is None else {"allocator_type": _allocator_type}
        _context = iris.iris(heap_size=size, **options)
    elif required > _context.heap_size:
        raise RuntimeError(
            f"Iris's {_context.heap_size}-byte heap cannot hold {required} bytes; "
            "configure the combined capacity before first preparation"
        )
    _reserved_bytes = required
    return _context


class IrisGroup:
    """Retain the context and the ordered heap addresses for one process group."""

    def __init__(self, ctx, group: dist.ProcessGroup, rank: int, device: torch.device):
        self.ctx = ctx
        self.group = group
        self.rank = rank
        self.device = device
        self.world_size = group.size()
        ranks = dist.get_process_group_ranks(group)
        assert ranks[rank] == dist.get_rank()
        self.bases = ctx.get_heap_bases()[ranks].contiguous()
        addresses = [int(address) for address in self.bases.tolist()]
        self.addresses = tuple(addresses + [addresses[-1]] * (8 - len(addresses)))

    def finish_prepare(self) -> None:
        """Publish initialized buffers before any peer can use them."""
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
