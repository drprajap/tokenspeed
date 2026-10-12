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

"""Compile the generic Iris all-reduce protocols for gfx1250 without launching."""

import re

import pytest
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_amd, reason="Iris requires AMD ROCm"
)

_GFX1250 = ("gfx1250", 32)


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def _aligned(fn, names):
    return {
        (index,): [["tt.divisibility", 16]]
        for index, name in enumerate(fn.arg_names)
        if name in names or name.startswith("heap_base_")
    }


def _compile(kind, world_size):
    from tokenspeed_kernel._triton import gl, gluon, triton
    from tokenspeed_kernel.ops.communication._iris import all_reduce as ar

    arch = ar.iris_arch_config(_GFX1250[0])
    if kind == "staged_oneshot":
        fn = ar.iris_pull_oneshot_staged
        signature = {
            "input_ptr": "*bf16",
            "input_sym_ptr": "*bf16",
            "output_ptr": "*bf16",
            "ready_flags": "*i32",
            "heap_bases": "*i64",
            "NUMEL": "i32",
        }
        constexprs = {
            "RANK": 0,
            "WORLD_SIZE": world_size,
            "BLOCK_SIZE": ar.StagedPullOneShot.BLOCK_SIZE,
            "SLOT_STRIDE": 16 * 7168,
            "NUM_SLOTS": ar.StagedPullOneShot.INPUT_SLOTS,
            "PEER_LOAD_CACHE": arch.peer_load_cache,
        }
        signature.update({name: "constexpr" for name in constexprs})
        source = triton.compiler.ASTSource(
            fn,
            signature,
            constexprs=constexprs,
            attrs=_aligned(fn, set(signature) - set(constexprs)),
        )
        num_warps = ar.StagedPullOneShot.NUM_SUBGROUPS
    else:
        two_shot = kind == "pull_twoshot"
        fn = ar.iris_pull_twoshot if two_shot else ar.iris_pull_oneshot
        pointers = (
            ("input_sym_ptr", "scratch_sym_ptr", "output_ptr")
            if two_shot
            else ("input_sym_ptr", "output_ptr")
        )
        signature = {name: "*bf16" for name in pointers}
        signature["ready_flags"] = "*i32"
        signature.update(
            {name: "i64" for name in fn.arg_names if name.startswith("heap_base_")}
        )
        sizes = ("PARTITION_WORDS" if two_shot else "TOTAL_NUMEL", "NUM_PROGRAMS")
        signature.update({name: "i32" for name in (*sizes, "NUM_TILES")})
        num_warps = 8 if two_shot else ar.ProducerPull.NUM_SUBGROUPS
        constexprs = {
            "RANK": 0,
            "WORLD_SIZE": world_size,
            "NUM_WARPS": num_warps,
            "SUBGROUP_SIZE": arch.subgroup_size,
            "WORDS_PER_LANE": 2,
            "ELEMENT_DTYPE": gl.bfloat16,
            "ELEMENTS_PER_WORD": 4,
            "PEER_LOAD_CACHE": arch.peer_load_cache,
            "REMOTE_STORE_CACHE": arch.remote_store_cache,
        }
        if two_shot:
            constexprs["BLOCK_WORDS"] = num_warps * arch.subgroup_size * 2 // world_size
            constexprs["EXIT_BARRIER"] = True
        else:
            constexprs["BLOCK_SIZE"] = ar.ProducerPull.BLOCK_SIZE
            constexprs["PUBLISH_READY"] = False
        source = gluon._runtime.GluonASTSource(
            fn,
            signature,
            constexprs=constexprs,
            attrs=_aligned(fn, (*pointers, "ready_flags")),
        )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", *_GFX1250),
        options={"num_warps": num_warps},
    )
    return kernel


def _backedge_cycles(lines):
    labels = {
        match[1]: index
        for index, line in enumerate(lines)
        if (match := re.match(r"^(\.LBB\w+):", line))
    }
    cycles = []
    for index, line in enumerate(lines):
        match = re.search(r"\bs_(?:cbranch_\w+|branch)\s+(\.LBB\w+)", line)
        if match and labels.get(match[1], index) < index:
            cycles.append((labels[match[1]], index))
    return cycles


# Peer flag polls per launch: one per peer for staged, one per rendezvous for
# the pull protocols (two-shot includes its exit barrier).
_EXPECTED_POLLS = {
    "staged_oneshot": lambda world_size: world_size - 1,
    "pull_oneshot": lambda world_size: 2,
    "pull_twoshot": lambda world_size: 3,
}


@pytest.mark.parametrize("world_size", (2, 4, 8))
@pytest.mark.parametrize("kind", tuple(_EXPECTED_POLLS))
def test_iris_all_reduce_codegen_gfx1250(kind, world_size, tmp_path):
    """Wave32 layouts compile, and only flag polls bypass the cache.

    A non-temporal peer data load refetches every partially read sector over
    the fabric on gfx1250, so the only non-temporal loads allowed are the
    system-scope flag polls, and each of them must stay on its retry backedge.
    """
    kernel = _compile(kind, world_size)
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"{kind}-ws{world_size}.amdgcn").write_text(assembly)
    lines = assembly.splitlines()

    loads = [line for line in lines if re.search(r"\b(global|buffer)_load_", line)]
    non_temporal = [line for line in loads if "TH_LOAD_NT" in line]
    assert all("SCOPE_SYS" in line for line in non_temporal), non_temporal

    polls = [
        index
        for index, line in enumerate(lines)
        if re.search(r"\b(global|buffer)_load_\w+\s.*SCOPE_SYS", line)
    ]
    assert len(polls) == _EXPECTED_POLLS[kind](world_size)
    cycles = _backedge_cycles(lines)
    assert all(
        any(start <= poll <= end for start, end in cycles) for poll in polls
    ), "a peer flag poll was hoisted out of its retry loop"

    if kind == "pull_twoshot":
        # The 32-bit global stores are the epochs written into peer flag rows.
        stores = [line for line in lines if re.search(r"\bglobal_store_b32", line)]
        assert stores
        assert not any("th:" in line or "scope:" in line for line in stores), stores
    assert any(
        re.search(r"\bglobal_atomic_\w+.*SCOPE_SYS", line) for line in lines
    ), "epochs must be published with system-scope atomics"
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


def test_iris_drain_waits_for_stores_gfx1250():
    """gfx1250 rejects ``vmcnt``; the drain must wait on loads and stores."""
    from tokenspeed_kernel._triton import gl, gluon, triton
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        _iris_drain_subgroup_vmem,
        iris_arch_config,
    )

    @gluon.jit
    def publish(flag, SPLIT_WAIT_COUNTERS: gl.constexpr):
        gl.store(flag, 1)
        _iris_drain_subgroup_vmem(SPLIT_WAIT_COUNTERS)
        gl.atomic_xchg(flag + 1, 1, sem="release", scope="sys")

    source = gluon._runtime.GluonASTSource(
        publish,
        {"flag": "*i32", "SPLIT_WAIT_COUNTERS": "constexpr"},
        constexprs={
            "SPLIT_WAIT_COUNTERS": iris_arch_config("gfx1250").split_wait_counters
        },
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", *_GFX1250),
        options={"num_warps": 1},
    )
    lines = kernel.asm["amdgcn"].splitlines()
    store = next(i for i, line in enumerate(lines) if "global_store_b32" in line)
    publish_at = next(
        i for i, line in enumerate(lines) if re.search(r"global_atomic_\w+", line)
    )
    between = lines[store:publish_at]
    assert any("s_wait_storecnt 0x0" in line for line in between)
    assert any("s_wait_loadcnt 0x0" in line for line in between)


def test_iris_arch_config_keeps_gfx950_codegen():
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        IrisArchConfig,
        iris_arch_config,
    )

    gfx950 = IrisArchConfig(
        subgroup_size=64,
        split_wait_counters=False,
        peer_load_cache=".cg",
        remote_store_cache=".wt",
    )
    assert iris_arch_config("gfx950") == gfx950
    assert iris_arch_config("gfx942") == gfx950
    assert iris_arch_config("gfx1250").subgroup_size == 32
