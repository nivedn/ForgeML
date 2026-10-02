"""Tests for forge_importer.import_module.

Exact-text golden tests pin the canonical models. Everything else is checked
with `_assert_valid_ir`, a small stand-in for the MLIR verifier (forge-opt does
not exist yet) that catches the bug class we hit before: a value declared with
one type and used as another, or a matmul whose shapes do not multiply.
"""

import re

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import forge_importer
from forge_importer import UnsupportedOpError, import_module

TOY_MLP_IR = """\
func.func @main(%arg0: tensor<8x10xf32>) -> tensor<8x32xf32> {
  %0 = forge.const dense<0.0> : tensor<32x10xf32>
  %1 = forge.const dense<0.0> : tensor<32xf32>
  %2 = forge.transpose %0 : (tensor<32x10xf32>) -> tensor<10x32xf32>
  %3 = forge.matmul %arg0, %2 : (tensor<8x10xf32>, tensor<10x32xf32>) -> tensor<8x32xf32>
  %4 = forge.add %3, %1 : tensor<8x32xf32>
  %5 = forge.relu %4 : tensor<8x32xf32>
  return %5 : tensor<8x32xf32>
}
"""

TWO_LAYER_IR = """\
func.func @main(%arg0: tensor<8x10xf32>) -> tensor<8x4xf32> {
  %0 = forge.const dense<0.0> : tensor<16x10xf32>
  %1 = forge.const dense<0.0> : tensor<16xf32>
  %2 = forge.const dense<0.0> : tensor<4x16xf32>
  %3 = forge.const dense<0.0> : tensor<4xf32>
  %4 = forge.transpose %0 : (tensor<16x10xf32>) -> tensor<10x16xf32>
  %5 = forge.matmul %arg0, %4 : (tensor<8x10xf32>, tensor<10x16xf32>) -> tensor<8x16xf32>
  %6 = forge.add %5, %1 : tensor<8x16xf32>
  %7 = forge.relu %6 : tensor<8x16xf32>
  %8 = forge.transpose %2 : (tensor<4x16xf32>) -> tensor<16x4xf32>
  %9 = forge.matmul %7, %8 : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
  %10 = forge.add %9, %3 : tensor<8x4xf32>
  return %10 : tensor<8x4xf32>
}
"""


# --- helpers ---------------------------------------------------------------


def _x(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(*shape).to(dtype)


class _Apply(nn.Module):
    """Wraps a function so any single op can be exported as a one-op model."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class _Identity(nn.Module):
    def forward(self, x):
        return x


class _BareMatmul(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(10, 4))

    def forward(self, x):
        return torch.mm(x, self.w)


class _TwoOutputs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 4)

    def forward(self, x):
        y = self.fc(x)
        return F.relu(y), y


_TENSOR_TYPE = r"tensor<[^>]*>"


def _dims(tensor_type: str) -> list[int]:
    m = re.fullmatch(r"tensor<((?:\d+x)*)(\w+)>", tensor_type)
    assert m, f"malformed tensor type: {tensor_type!r}"
    return [int(d) for d in m.group(1).split("x") if d]


def _assert_valid_ir(text: str) -> None:
    lines = text.rstrip("\n").splitlines()
    header = re.fullmatch(
        rf"func\.func @main\(%arg0: ({_TENSOR_TYPE})\) -> ({_TENSOR_TYPE}) \{{",
        lines[0],
    )
    assert header, f"bad function header: {lines[0]!r}"
    assert lines[-1] == "}", "function body is not closed"

    types = {"%arg0": header.group(1)}
    return_type = header.group(2)
    saw_return = False

    def type_of(name: str) -> str:
        assert name in types, f"{name} used before it is defined"
        return types[name]

    for raw in lines[1:-1]:
        line = raw.strip()

        ret = re.fullmatch(rf"return (%\w+) : ({_TENSOR_TYPE})", line)
        if ret:
            assert type_of(ret.group(1)) == ret.group(2) == return_type
            saw_return = True
            continue

        m = re.fullmatch(r"(%\w+) = forge\.(\w+) (.+)", line)
        assert m, f"unparseable line: {line!r}"
        name, op, rest = m.groups()
        assert name not in types, f"{name} is defined twice"

        if op == "const":
            c = re.fullmatch(rf"dense<[^>]*> : ({_TENSOR_TYPE})", rest)
            assert c, f"bad const: {line!r}"
            types[name] = c.group(1)
        elif op in ("transpose", "matmul"):
            s = re.fullmatch(rf"(.+?) : \((.+?)\) -> ({_TENSOR_TYPE})", rest)
            assert s, f"bad {op}: {line!r}"
            operands = s.group(1).split(", ")
            in_types = s.group(2).split(", ")
            out_type = s.group(3)
            assert [type_of(o) for o in operands] == in_types, (
                f"operand types disagree with their definitions: {line!r}"
            )
            if op == "transpose":
                assert len(operands) == 1
                assert _dims(in_types[0])[::-1] == _dims(out_type)
            else:
                assert len(operands) == 2
                lhs, rhs, out = map(_dims, (in_types[0], in_types[1], out_type))
                assert lhs[-1] == rhs[-2], f"inner dims do not agree: {line!r}"
                assert out == lhs[:-1] + [rhs[-1]], f"bad result shape: {line!r}"
            types[name] = out_type
        elif op == "add":
            a = re.fullmatch(rf"(.+?) : ({_TENSOR_TYPE})", rest)
            assert a, f"bad add: {line!r}"
            lhs_name, rhs_name = a.group(1).split(", ")
            result = a.group(2)
            assert type_of(lhs_name) == result
            rhs_dims, result_dims = _dims(type_of(rhs_name)), _dims(result)
            assert result_dims[len(result_dims) - len(rhs_dims) :] == rhs_dims, (
                f"add operand is not broadcast-compatible: {line!r}"
            )
            types[name] = result
        elif op == "relu":
            r = re.fullmatch(rf"(%\w+) : ({_TENSOR_TYPE})", rest)
            assert r, f"bad relu: {line!r}"
            assert type_of(r.group(1)) == r.group(2)
            types[name] = r.group(2)
        else:
            pytest.fail(f"unknown forge op: {op}")

    assert saw_return, "function never returns"


# --- golden output ---------------------------------------------------------


def test_single_linear_relu_matches_golden_output():
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
    assert import_module(model, _x(8, 10)) == TOY_MLP_IR


def test_two_layer_mlp_matches_golden_output():
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 4))
    assert import_module(model, _x(8, 10)) == TWO_LAYER_IR


# --- supported ops: structure holds across shapes --------------------------


@pytest.mark.parametrize(
    ("input_shape", "out_features"),
    [
        pytest.param((1, 10), 32, id="batch-of-one"),
        pytest.param((8, 10), 32, id="wide"),
        pytest.param((8, 32), 10, id="narrow"),
        pytest.param((8, 10), 10, id="square-weight"),
        pytest.param((3, 1), 1, id="1x1"),
        pytest.param((2, 8, 10), 4, id="3d-input"),
    ],
)
def test_linear_is_well_formed_for_any_shape(input_shape, out_features):
    model = nn.Linear(input_shape[-1], out_features)
    text = import_module(model, _x(*input_shape))
    _assert_valid_ir(text)
    # aten.linear always means x @ W^T, even when W happens to be square
    assert "forge.transpose" in text


def test_linear_without_relu_returns_the_bias_add():
    text = import_module(nn.Linear(10, 4), _x(8, 10))
    _assert_valid_ir(text)
    assert "forge.relu" not in text
    op_before_return = text.strip().splitlines()[-3]
    assert "forge.add" in op_before_return


def test_relu_only_model_has_no_constants():
    text = import_module(_Apply(F.relu), _x(4, 4))
    _assert_valid_ir(text)
    assert "forge.const" not in text


def test_identity_model_returns_the_argument():
    text = import_module(_Identity(), _x(8, 10))
    _assert_valid_ir(text)
    assert "return %arg0" in text


def test_bare_matmul_is_not_transposed():
    text = import_module(_BareMatmul(), _x(8, 10))
    _assert_valid_ir(text)
    assert "forge.transpose" not in text


def test_add_of_input_with_itself():
    text = import_module(_Apply(lambda t: t + t), _x(8, 10))
    _assert_valid_ir(text)
    assert "forge.add %arg0, %arg0" in text


# --- fail loudly -----------------------------------------------------------


@pytest.mark.parametrize(
    ("make_model", "input_shape", "op_name"),
    [
        pytest.param(lambda: _Apply(torch.sigmoid), (8, 10), "sigmoid", id="sigmoid"),
        pytest.param(lambda: _Apply(torch.tanh), (8, 10), "tanh", id="tanh"),
        pytest.param(lambda: _Apply(F.gelu), (8, 10), "gelu", id="gelu"),
        pytest.param(
            lambda: _Apply(lambda t: F.softmax(t, dim=-1)),
            (8, 10),
            "softmax",
            id="softmax",
        ),
        pytest.param(lambda: nn.Conv2d(1, 2, 3), (1, 1, 8, 8), "conv2d", id="conv2d"),
    ],
)
def test_unsupported_op_raises_and_names_the_op(make_model, input_shape, op_name):
    with pytest.raises(UnsupportedOpError, match=op_name):
        import_module(make_model(), _x(*input_shape))


def test_unsupported_op_error_message_names_the_full_overload():
    with pytest.raises(UnsupportedOpError, match=r"aten\.sigmoid\.default"):
        import_module(_Apply(torch.sigmoid), _x(8, 10))


def test_unsupported_op_error_is_a_not_implemented_error():
    assert issubclass(UnsupportedOpError, NotImplementedError)


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float16, torch.bfloat16, torch.int64]
)
def test_unsupported_input_dtype_raises(dtype):
    with pytest.raises(UnsupportedOpError, match="dtype"):
        import_module(_Identity(), _x(8, 10, dtype=dtype))


def test_unsupported_parameter_dtype_raises():
    model = nn.Linear(10, 4).double()
    with pytest.raises(UnsupportedOpError, match="dtype"):
        import_module(model, _x(8, 10, dtype=torch.float64))


# --- known gaps: these fail until the importer is fixed --------------------


def test_linear_without_bias_never_crashes_unexpectedly():
    """bias=False gives aten.linear two args, not three. Either supporting it
    or raising UnsupportedOpError is fine; a raw IndexError is not."""
    model = nn.Linear(10, 4, bias=False)
    try:
        text = import_module(model, _x(8, 10))
    except UnsupportedOpError:
        return
    _assert_valid_ir(text)
    assert "forge.matmul" in text
    assert "forge.add" not in text


def test_multiple_outputs_are_rejected_not_truncated():
    """Single-output is a stated scope limit. Returning only the first of two
    outputs would silently produce IR for a different function."""
    with pytest.raises(UnsupportedOpError):
        import_module(_TwoOutputs(), _x(8, 10))


# --- importer hygiene ------------------------------------------------------


def test_public_api():
    assert {"import_module", "UnsupportedOpError"} <= set(forge_importer.__all__)


def test_repeated_imports_of_the_same_model_are_identical():
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
    assert import_module(model, _x(8, 10)) == import_module(model, _x(8, 10))


def test_output_depends_on_input_shape_not_input_values():
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
    assert import_module(model, torch.zeros(8, 10)) == import_module(
        model, torch.randn(8, 10) * 1000
    )
