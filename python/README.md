# forge-importer

Turns a PyTorch model into ForgeML's own IR, written as `forge`-dialect MLIR text.

It traces the model with `torch.export`, walks the resulting ATen graph one node at a time, and emits one or more `forge.*` lines per node. Anything it does not recognise raises an error instead of being guessed at.

**Status: Phase 1.** The IR has the right structure and types, but every weight is a placeholder zero and the `forge` dialect is not defined in C++ yet. See [Known limits](#known-limits).

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd python
uv sync
```

```python
import torch
from torch import nn
from forge_importer import import_module

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
print(import_module(model, torch.randn(8, 10)))
```

```mlir
func.func @main(%arg0: tensor<8x10xf32>) -> tensor<8x32xf32> {
  %0 = forge.const dense<0.0> : tensor<32x10xf32>
  %1 = forge.const dense<0.0> : tensor<32xf32>
  %2 = forge.transpose %0 : (tensor<32x10xf32>) -> tensor<10x32xf32>
  %3 = forge.matmul %arg0, %2 : (tensor<8x10xf32>, tensor<10x32xf32>) -> tensor<8x32xf32>
  %4 = forge.add %3, %1 : tensor<8x32xf32>
  %5 = forge.relu %4 : tensor<8x32xf32>
  return %5 : tensor<8x32xf32>
}
```

The public API is two names:

| Name | What it is |
|---|---|
| `import_module(model, example_input) -> str` | Returns the MLIR text. The example input is run through the model once to trace it; only its shape and dtype matter, not its values. |
| `UnsupportedOpError` | A `NotImplementedError` raised for anything outside the supported subset. |

### Running an example file

From the repository root:

```bash
uv run --project python python examples/run_importer.py examples/toy_mlp.py
```

The runner loads the `.py` file, instantiates the first `nn.Module` class defined in it, feeds it a fixed `torch.zeros(8, 10)` input, and prints the model, its output shapes, and the MLIR. The examples are `toy_mlp.py` (Linear + ReLU), `linear.py` (Linear without bias) and `mlp_two_output.py` (two outputs).

## What it supports

**Model shape:** one tensor input, one or more tensor outputs, `float32` only, static shapes (the example input's shape is fixed into the IR).

**Operators:**

| PyTorch | ATen op | Emitted as |
|---|---|---|
| `nn.Linear`, `F.linear` | `aten.linear` | `forge.transpose` of the weight, `forge.matmul`, then `forge.add` for the bias if there is one |
| `torch.mm` | `aten.mm` | `forge.matmul` (no transpose) |
| `a + b`, two tensors | `aten.add.Tensor` | `forge.add` |
| `nn.ReLU`, `F.relu` | `aten.relu` | `forge.relu` |
| any parameter or buffer | graph input | `forge.const dense<0.0>` with the parameter's own shape |

`aten.linear` always transposes its weight, because PyTorch stores a Linear weight as `[out, in]` and computes `x @ W^T`. This is why a square weight still gets a `forge.transpose`.

**Output format:**

- The function is `@main`. The model input is always `%arg0`.
- One result is declared bare (`-> T`). Several are parenthesised (`-> (T1, T2)`).
- `return` lists the SSA values in the order the model returns them, so output 0 of the model is result 0 of the function.

```mlir
func.func @main(%arg0: tensor<8x10xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>) {
  ...
  return %5, %4 : tensor<8x4xf32>, tensor<8x4xf32>
}
```

## Errors

The importer never approximates. It raises `UnsupportedOpError`, with a message that names the problem:

| Cause | Message |
|---|---|
| An op not in the table above | `aten.sigmoid.default` |
| A dtype other than `float32` | `unsupported dtype: torch.float64` |
| A scalar where a tensor is expected, e.g. `x + 5` | `expected a tensor operand, got 5` |
| A returned value that is not a tensor | `non-tensor output: None` |

## Known limits

- **Weights are placeholders.** Every parameter and buffer becomes `dense<0.0>`. The IR proves structure and shapes, not numbers, so it cannot yet be used to check numerical correctness. Real weights are planned for a later phase.
- **Text only, and not yet verified by MLIR.** The `forge` dialect does not exist in C++ yet, so `forge.*` ops are defined only by this importer. Nothing parses the output until `forge-opt` exists. The tests use a small structural checker as a stand-in.
- **Narrow scope:** `float32`, a single input, static shapes, no control flow, and only the four operators in the table.
- **Dict and nested outputs are flattened** into a positional tuple, so key names are lost.
- **Inputs with more than two dimensions** are emitted as written. For `aten.linear` with a 3-D input the matmul line is well-formed, but whether `forge.matmul` means a batched matmul is undecided until the dialect is defined.

## Development

```bash
cd python
uv run pytest
uv run ruff check src tests
uv run mypy src
```

The code is in `src/forge_importer/importer.py`: a dispatch table maps each ATen op to a small `emit_*` function that appends IR lines and records the SSA name of its result. Adding an operator means adding one emitter and one table entry.

Tests live in `tests/test_importer.py`: exact-text goldens for the canonical models, a structural validity check for everything else, and failure-path tests for each error above.
