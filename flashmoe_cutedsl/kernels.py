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

    @cute.kernel
    def _gated_up_kernel(
        expert_in: cute.Tensor,
        up: cute.Tensor,
        bias_up: cute.Tensor,
        up_v: cute.Tensor,
        bias_up_v: cute.Tensor,
        hidden_out: cute.Tensor,
        total: cutlass.Int32,
        experts: cutlass.Int32,
        hidden_dim: cutlass.Int32,
        ffn_dim: cutlass.Int32,
        capacity: cutlass.Int32,
        act_type: cutlass.Constexpr[int],
        swish_alpha: cutlass.Float32,
        swish_beta: cutlass.Float32,
        block_cols: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        n = bidx * block_cols + tidx
        slot = bidy
        expert = bidz
        if n < ffn_dim:
            count = total // hidden_dim
            row = expert * capacity + slot
            acc = cutlass.Float32(0.0)
            acc_v = cutlass.Float32(0.0)
            k = cutlass.Int32(0)
            while k < hidden_dim:
                x = cutlass.Float32(expert_in[expert, slot, k])
                acc += x * cutlass.Float32(up[expert, n, k])
                acc_v += x * cutlass.Float32(up_v[expert, n, k])
                k += 1
            acc += cutlass.Float32(bias_up[expert, n])
            acc_v += cutlass.Float32(bias_up_v[expert, n])
            gate_x = swish_beta * acc
            gate = swish_alpha * (gate_x / (cutlass.Float32(1.0) + cute.exp(-gate_x, fastmath=True)))
            if row < count:
                hidden_out[expert, slot, n] = (gate * acc_v).to(hidden_out.element_type)

    @cute.kernel
    def _linear_down_kernel(
        hidden_in: cute.Tensor,
        down: cute.Tensor,
        bias_down: cute.Tensor,
        expert_out: cute.Tensor,
        total: cutlass.Int32,
        hidden_dim: cutlass.Int32,
        ffn_dim: cutlass.Int32,
        capacity: cutlass.Int32,
        block_cols: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        col = bidx * block_cols + tidx
        slot = bidy
        expert = bidz
        if col < hidden_dim:
            count = total // hidden_dim
            row = expert * capacity + slot
            acc = cutlass.Float32(0.0)
            k = cutlass.Int32(0)
            while k < ffn_dim:
                acc += cutlass.Float32(hidden_in[expert, slot, k]) * cutlass.Float32(down[expert, col, k])
                k += 1
            acc += cutlass.Float32(bias_down[expert, col])
            if row < count:
                expert_out[expert, slot, col] = acc.to(expert_out.element_type)

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

    @cute.jit
    def _launch_gated_up(
        expert_in: cute.Tensor,
        up: cute.Tensor,
        bias_up: cute.Tensor,
        up_v: cute.Tensor,
        bias_up_v: cute.Tensor,
        hidden_out: cute.Tensor,
        total: cutlass.Int32,
        experts: cutlass.Int32,
        hidden_dim: cutlass.Int32,
        ffn_dim: cutlass.Int32,
        capacity: cutlass.Int32,
        act_type: cutlass.Constexpr[int],
        swish_alpha: cutlass.Float32,
        swish_beta: cutlass.Float32,
        block: cutlass.Constexpr[int],
    ):
        grid_x = cute.ceil_div(ffn_dim, block)
        _gated_up_kernel(
            expert_in,
            up,
            bias_up,
            up_v,
            bias_up_v,
            hidden_out,
            total,
            experts,
            hidden_dim,
            ffn_dim,
            capacity,
            act_type,
            swish_alpha,
            swish_beta,
            block,
        ).launch(
            grid=(grid_x, capacity, experts),
            block=(block, 1, 1),
        )

    @cute.jit
    def _launch_linear_down(
        hidden_in: cute.Tensor,
        down: cute.Tensor,
        bias_down: cute.Tensor,
        expert_out: cute.Tensor,
        total: cutlass.Int32,
        experts: cutlass.Int32,
        hidden_dim: cutlass.Int32,
        ffn_dim: cutlass.Int32,
        capacity: cutlass.Int32,
        block: cutlass.Constexpr[int],
    ):
        grid_x = cute.ceil_div(hidden_dim, block)
        _linear_down_kernel(
            hidden_in,
            down,
            bias_down,
            expert_out,
            total,
            hidden_dim,
            ffn_dim,
            capacity,
            block,
        ).launch(
            grid=(grid_x, capacity, experts),
            block=(block, 1, 1),
        )

    return _launch_gather_tokens, _launch_combine_top1, _launch_gated_up, _launch_linear_down


@lru_cache(maxsize=1)
def _combine_top1_launcher():
    return _define_kernels()[1]


def _gather_tokens_launcher():
    return _define_kernels()[0]


@lru_cache(maxsize=1)
def _gated_up_launcher():
    return _define_kernels()[2]


@lru_cache(maxsize=1)
def _linear_down_launcher():
    return _define_kernels()[3]


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


def gated_up(expert_in, weights, hidden_out, *, act_type: int, swish_alpha: float = 1.0, swish_beta: float = 1.0, stream=None, block: int = 256) -> None:
    """CuTe DSL fused gated up/value projection.

    Computes ``activation(expert_in @ up.T + bias_up) * (expert_in @ up_v.T + bias_up_v)``
    for tensors laid out as ``[experts, capacity, dim]``.
    """
    cutlass, _, from_dlpack = _imports()
    if weights.local_expert_up_v is None or weights.local_bias_up_v is None:
        raise ValueError("gated_up requires up_v and bias_up_v weights")
    if expert_in.ndim != 3:
        raise ValueError("expert_in must have shape [experts, capacity, hidden]")
    experts, capacity, hidden = expert_in.shape
    if weights.local_expert_up.shape[0] != experts:
        raise ValueError("weight expert dimension does not match expert_in")
    ffn = int(weights.local_expert_up.shape[1])
    if hidden_out.shape != (experts, capacity, ffn):
        raise ValueError("hidden_out must have shape [experts, capacity, ffn]")

    _gated_up_launcher()(
        from_dlpack(expert_in),
        from_dlpack(weights.local_expert_up),
        from_dlpack(weights.local_bias_up),
        from_dlpack(weights.local_expert_up_v),
        from_dlpack(weights.local_bias_up_v),
        from_dlpack(hidden_out),
        cutlass.Int32(int(expert_in.numel())),
        cutlass.Int32(int(experts)),
        cutlass.Int32(int(hidden)),
        cutlass.Int32(ffn),
        cutlass.Int32(int(capacity)),
        int(act_type),
        cutlass.Float32(float(swish_alpha)),
        cutlass.Float32(float(swish_beta)),
        block,
    )


def linear_down(hidden_in, weights, expert_out, *, stream=None, block: int = 256) -> None:
    """CuTe DSL down projection for packed expert outputs."""
    cutlass, _, from_dlpack = _imports()
    if hidden_in.ndim != 3:
        raise ValueError("hidden_in must have shape [experts, capacity, ffn]")
    experts, capacity, ffn = hidden_in.shape
    hidden = int(weights.local_expert_down.shape[1])
    if expert_out.shape != (experts, capacity, hidden):
        raise ValueError("expert_out must have shape [experts, capacity, hidden]")

    _linear_down_launcher()(
        from_dlpack(hidden_in),
        from_dlpack(weights.local_expert_down),
        from_dlpack(weights.local_bias_down),
        from_dlpack(expert_out),
        cutlass.Int32(int(expert_out.numel())),
        cutlass.Int32(int(experts)),
        cutlass.Int32(hidden),
        cutlass.Int32(int(ffn)),
        cutlass.Int32(int(capacity)),
        block,
    )
