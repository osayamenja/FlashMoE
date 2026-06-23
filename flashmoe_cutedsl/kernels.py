from __future__ import annotations

from functools import lru_cache


def _imports():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    return cutlass, cute, from_dlpack


def _define_kernels():
    cutlass, cute, _ = _imports()

    @cute.kernel
    def _gather_tokens_kernel(tokens: cute.Tensor, token_ids: cute.Tensor, expert_in: cute.Tensor, total: cutlass.Int32, hidden: cutlass.Int32, capacity: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        elem = bidx * bdim + tidx
        if elem < total:
            row = elem // hidden
            col = elem - row * hidden
            expert = row // capacity
            slot = row - expert * capacity
            token = token_ids[expert, slot]
            if token >= 0:
                expert_in[expert, slot, col] = tokens[token, col]
            else:
                z = tokens[0, col]
                expert_in[expert, slot, col] = z - z

    @cute.kernel
    def _combine_top1_kernel(values: cute.Tensor, token_ids: cute.Tensor, output: cute.Tensor, total: cutlass.Int32, hidden: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        elem = bidx * bdim + tidx
        if elem < total:
            row = elem // hidden
            col = elem - row * hidden
            token = token_ids[row]
            if token >= 0:
                output[token, col] = values[row, col]

    @cute.jit
    def _launch_gather_tokens(tokens: cute.Tensor, token_ids: cute.Tensor, expert_in: cute.Tensor, total: cutlass.Int32, hidden: cutlass.Int32, capacity: cutlass.Int32, block: cutlass.Constexpr[int]):
        grid_x = cute.ceil_div(total, block)
        _gather_tokens_kernel(tokens, token_ids, expert_in, total, hidden, capacity).launch(
            grid=(grid_x, 1, 1),
            block=(block, 1, 1),
        )

    @cute.jit
    def _launch_combine_top1(values: cute.Tensor, token_ids: cute.Tensor, output: cute.Tensor, total: cutlass.Int32, hidden: cutlass.Int32, block: cutlass.Constexpr[int]):
        grid_x = cute.ceil_div(total, block)
        _combine_top1_kernel(values, token_ids, output, total, hidden).launch(
            grid=(grid_x, 1, 1),
            block=(block, 1, 1),
        )

    return _launch_gather_tokens, _launch_combine_top1


@lru_cache(maxsize=1)
def _combine_top1_launcher():
    return _define_kernels()[1]


def _gather_tokens_launcher():
    return _define_kernels()[0]


def gather_tokens(tokens, token_ids, expert_in, *, stream=None, block: int = 256) -> None:
    """CuTe DSL gather from `[S, H]` into `[E, capacity, H]`."""
    cutlass, _, from_dlpack = _imports()
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [sequence, hidden]")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [experts, capacity]")
    if expert_in.shape != (*token_ids.shape, tokens.shape[1]):
        raise ValueError("expert_in must have shape [experts, capacity, hidden]")

    total = int(expert_in.numel())
    hidden = int(tokens.shape[1])
    capacity = int(token_ids.shape[1])
    launcher = _gather_tokens_launcher()
    launcher(
        from_dlpack(tokens),
        from_dlpack(token_ids),
        from_dlpack(expert_in),
        cutlass.Int32(total),
        cutlass.Int32(hidden),
        cutlass.Int32(capacity),
        block,
    )


def combine_top1(values, token_ids, output, *, stream=None, block: int = 256) -> None:
    """CuTe DSL combine helper for the top-1, no-reduction case.

    ``values`` is ``[N, H]``, ``token_ids`` is ``[N]``, and ``output`` is
    ``[S, H]``.  Top-k reductions use the torch reference path for now because
    they require atomic accumulation.
    """
    cutlass, _, from_dlpack = _imports()
    if values.ndim != 2:
        raise ValueError("values must have shape [tokens, hidden]")
    if token_ids.ndim != 1 or token_ids.shape[0] != values.shape[0]:
        raise ValueError("token_ids must have shape [tokens]")
    if output.ndim != 2 or output.shape[1] != values.shape[1]:
        raise ValueError("output must have shape [sequence, hidden]")

    total = int(values.numel())
    hidden = int(values.shape[1])
    launcher = _combine_top1_launcher()
    launcher(
        from_dlpack(values),
        from_dlpack(token_ids),
        from_dlpack(output),
        cutlass.Int32(total),
        cutlass.Int32(hidden),
        block,
    )
