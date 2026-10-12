# Iris communication

`IrisBackend` retains a prepared `IrisAllReduce` for each process group and dtype.
The operation directly owns its staging, producer, scratch, and synchronization
buffers. All operations share one process-wide Iris context and symmetric heap;
there is no capacity-based workspace lookup during execution.

Preparation must run in the same order on participating ranks, before cache
sizing and graph capture. Capacities cannot grow once prepared. Kimi-K3 prepares
its combined requirements through `prepare_communication_runtime`, retains the
fusion workspace on the model, and shares it across sequential layers and the
NextN draft. Startup binds each caller directly to the same group workspace.
Other models retain the existing lazy initialization during warmup.

Generic all-reduce selection depends on the architecture, group size, dtype,
and payload size. Exact-shape tuning uses the same protocol with its own launch
geometry. Different staged geometries need separate slots and epochs: their
workgroup IDs cover different element ranges. Producer-direct reduction has its
own input and completion protocol, so staging cannot overwrite producer inputs.

K3's Lamport window, AttnRes, and attention/MoE row-sharded fusions live in
`iris_kimi_k3.py`, outside the generic backend. Its workspace explicitly borrows
the generic producer and pull scratch. Producers join before communication;
operations using those buffers must be ordered on one stream. Sharded fusions
return a borrowed activation valid until the next fusion; ordinary
producer-direct all-reduce returns owned outputs. Local residual shards remain
owned by the caller.

Add a collective or fusion as an operation with its own required storage and
explicit prepared handle. Reuse the context and group addressing; share scratch
only when the operations' ordering permits it. The runtime's Triton RSAG path
remains separate, and `AutoBackend` retains its topology and numerics policy.

To select an Iris allocator or a larger initial heap, call this before any Iris
preparation on every rank:

```python
from tokenspeed_kernel.ops.communication.iris import configure_iris

configure_iris(heap_size=None, allocator_type="torch")
```

`None` keeps automatic heap sizing or Iris's default allocator. Allocator names
are forwarded unchanged; Iris's `IRIS_ALLOCATOR` override keeps precedence.
The pinned Iris supports `torch` and `vmem`; newer allocator choices become
available with the corresponding Iris version. A live context is never replaced.
