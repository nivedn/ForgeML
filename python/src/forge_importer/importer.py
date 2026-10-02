"""Translates a supported PyTorch subset into forge-dialect MLIR text.

Supported subset (Phase 1): single-input, single-output models built from
nn.Linear and ReLU. Anything else raises UnsupportedOpError rather than
silently producing wrong IR.
"""

from dataclasses import dataclass, field

import torch
from torch.export import export


class UnsupportedOpError(NotImplementedError):
    """Raised when the exported graph uses an op outside the supported subset."""


@dataclass
class EmitContext:
    """Shared state threaded through the node walk."""

    value_names: dict[str, str] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)
    _counter: int = 0

    def fresh_ssa(self) -> str:
        ssa = self._counter
        self._counter += 1
        return f"%{ssa}"


def emit_matmul(node: torch.fx.Node, ctx: EmitContext) -> None:

    args = node.args
    lhs_node = args[0]
    rhs_node = args[1]

    ssa_matmul = ctx.fresh_ssa()
    type_lhs_node = tensor_type(lhs_node.meta["val"].shape, lhs_node.meta["val"].dtype)
    type_rhs_node = tensor_type(rhs_node.meta["val"].shape, rhs_node.meta["val"].dtype)
    type_result_node = tensor_type(node.meta["val"].shape, node.meta["val"].dtype)
    signature = f"({type_lhs_node}, {type_rhs_node}) -> {type_result_node}"
    ctx.lines.append(
        f"{ssa_matmul} = forge.matmul {ctx.value_names[args[0].name]}, {ctx.value_names[args[1].name]} : {signature}"
    )

    ctx.value_names[node.name] = ssa_matmul


def emit_add(node: torch.fx.Node, ctx: EmitContext) -> None:
    args = node.args

    ssa_add = ctx.fresh_ssa()
    type_str_add = tensor_type(node.meta["val"].shape, node.meta["val"].dtype)
    ctx.lines.append(
        f"{ssa_add} = forge.add {ctx.value_names[args[0].name]}, {ctx.value_names[args[1].name]} : {type_str_add}"
    )

    ctx.value_names[node.name] = ssa_add


def emit_linear(node: torch.fx.Node, ctx: EmitContext) -> None:
    args = node.args

    ## matmul
    lhs_node = args[0]
    rhs_node = args[1]

    type_lhs_node = tensor_type(lhs_node.meta["val"].shape, lhs_node.meta["val"].dtype)
    type_rhs_node = tensor_type(rhs_node.meta["val"].shape, rhs_node.meta["val"].dtype)

    ### Transpose Op
    transpose_rhs_node = torch.transpose(rhs_node.meta["val"], 0, 1)
    type_rhs_node_transpose = tensor_type(
        transpose_rhs_node.shape, rhs_node.meta["val"].dtype
    )
    ssa_transpose = ctx.fresh_ssa()
    ctx.lines.append(
        f"{ssa_transpose} = forge.transpose {ctx.value_names[args[1].name]} : ({type_rhs_node}) -> {type_rhs_node_transpose}"
    )
    ctx.value_names["transpose"] = ssa_transpose

    ssa_matmul = ctx.fresh_ssa()
    type_result_node = tensor_type(node.meta["val"].shape, node.meta["val"].dtype)
    signature = f"({type_lhs_node}, {type_rhs_node_transpose}) -> {type_result_node}"
    ctx.lines.append(
        f"{ssa_matmul} = forge.matmul {ctx.value_names[args[0].name]}, {ssa_transpose} : {signature}"
    )

    ## add
    ssa_add = ctx.fresh_ssa()
    type_str_add = tensor_type(node.meta["val"].shape, node.meta["val"].dtype)
    ctx.lines.append(
        f"{ssa_add} = forge.add {ssa_matmul}, {ctx.value_names[args[2].name]} : {type_str_add}"
    )

    ctx.value_names[node.name] = ssa_add


def emit_relu(node: torch.fx.Node, ctx: EmitContext) -> None:
    args = node.args

    ssa_relu = ctx.fresh_ssa()
    type_str_add = tensor_type(node.meta["val"].shape, node.meta["val"].dtype)
    ctx.lines.append(
        f"{ssa_relu} = forge.relu {ctx.value_names[args[0].name]} : {type_str_add}"
    )

    ctx.value_names[node.name] = ssa_relu


DISPATCH_TABLE = {
    torch.ops.aten.mm.default: emit_matmul,
    torch.ops.aten.add.Tensor: emit_add,
    torch.ops.aten.relu.default: emit_relu,
    torch.ops.aten.linear.default: emit_linear,
}

DTYPE_MAP = {torch.float32: "f32"}


def tensor_type(shape: torch.Size, dtype: torch.dtype) -> str:
    dtype_str = DTYPE_MAP.get(dtype)
    if dtype_str is None:
        raise UnsupportedOpError(f"unsupported dtype: {dtype}")
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}x{dtype_str}>"


def import_module(model: torch.nn.Module, example_input: torch.Tensor) -> str:
    """Traces `model` and returns it as forge-dialect MLIR text."""
    exported = export(model, (example_input,))
    ctx = EmitContext()

    for node in exported.graph.nodes:
        # print(f"node {node.op} = {node}")
        if node.op == "placeholder":
            if str(node) in exported.graph_signature.user_inputs:
                ctx.value_names[node.name] = "%arg0"
                input_nodes_type = tensor_type(
                    node.meta["val"].shape, node.meta["val"].dtype
                )
                continue

            ssa = ctx.fresh_ssa()
            type_str = tensor_type(node.meta["val"].shape, node.meta["val"].dtype)
            ctx.lines.append(f"{ssa} = forge.const dense<0.0> : {type_str}")
            ctx.value_names[node.name] = f"{ssa}"

        elif node.op == "call_function":
            if node.target not in DISPATCH_TABLE:
                raise UnsupportedOpError(node.target)

            emit_func = DISPATCH_TABLE[node.target]
            emit_func(node, ctx)

        elif node.op == "output":
            output_node = node.args[0][0]
            type_output = tensor_type(
                output_node.meta["val"].shape, output_node.meta["val"].dtype
            )
            ctx.lines.append(
                f"return {ctx.value_names[output_node.name]} : {type_output}"
            )

    body = "\n".join(f"  {line}" for line in ctx.lines)
    mlir_text = (
        f"func.func @main(%arg0: {input_nodes_type}) -> {type_output} {{\n{body}\n}}\n"
    )
    return mlir_text
