"""Translates a supported PyTorch subset into forge-dialect MLIR text.

Supported subset (Phase 1): single-input, single-output models built from
nn.Linear and ReLU. Anything else raises UnsupportedOpError rather than
silently producing wrong IR.
"""

from dataclasses import dataclass, field
import sys
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
        # TODO: allocate and return the next "%N" name, bump the counter
        ...


def emit_matmul(node, ctx: EmitContext) -> None:
    # TODO: look up operand SSA names from ctx.value_names via node.args,
    # read the output type from node.meta["val"], append the forge.matmul
    # line, record the result in ctx.value_names
    ...


def emit_add(node, ctx: EmitContext) -> None:
    # TODO: same shape as emit_matmul, for forge.add
    ...


def emit_relu(node, ctx: EmitContext) -> None:
    # TODO: same shape, single operand, for forge.relu
    ...


# TODO: confirm the exact op overload names (e.g. torch.ops.aten.mm.default)
# by printing node.target while walking a real traced graph — overload
# suffixes aren't always what you'd guess.
DISPATCH_TABLE = {
    # torch.ops.aten.mm.default: emit_matmul,
    # torch.ops.aten.add.Tensor: emit_add,
    # torch.ops.aten.relu.default: emit_relu,
}


def import_module(model: torch.nn.Module, example_input: torch.Tensor) -> str:
    """Traces `model` and returns it as forge-dialect MLIR text."""
    exported = export(model, (example_input,))
    print(exported)
    ctx = EmitContext()

    for node in exported.graph.nodes:
        if node.op == "placeholder":
            # TODO: this fires for BOTH the user's input and the model's
            # own weight/bias parameters — torch.export represents both as
            # placeholder nodes. You'll need exported.graph_signature to
            # tell them apart: the user input becomes %arg0; parameter
            # placeholders need a forge.const emitted for them (zeros, for
            # Phase 1) before anything downstream can reference them.
            ...
        elif node.op == "call_function":
            # TODO: dispatch-table lookup; raise UnsupportedOpError(node.target)
            # on a miss, no default/fallback case
            ...
        elif node.op == "output":
            # TODO: emit the func.func's `return`, using ctx.value_names
            ...

    # TODO: wrap ctx.lines in the `func.func @main(...) -> ... { ... }` shell
    ...

