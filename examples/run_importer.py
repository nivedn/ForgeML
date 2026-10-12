"""Runs the Forge importer against a chosen example model, prints the result.

Usage:
    python3 run_importer.py <model_module>

<model_module> is a sibling module (e.g. "toy_mlp") that defines, by
convention:
    build_model() -> torch.nn.Module
    build_input() -> torch.Tensor
"""

import importlib.util
import inspect
import sys

import torch
from forge_importer import import_module
from torch import nn


def main(model_file) -> None:

    model_classes = {}
    model_class_obj_list = []
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    if spec is None or spec.loader is None:
        raise SystemExit(
            f"cannot load {model_file!r} as a Python module (expected a .py file)"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the classes which are nn.Module type
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            obj.__module__ == module.__name__
            and issubclass(obj, nn.Module)
            and obj is not nn.Module
        ):
            print(name)
            print(obj)
            model_classes[name] = obj
            model_class_obj_list.append(obj)

    main_module_obj = model_class_obj_list[0]

    model = main_module_obj()
    example_input = torch.zeros(8, 10)
    output = model(example_input)

    # Print the model stats
    print(model)
    print("Input shape :", example_input.shape)
    if isinstance(output, (tuple, list)):
        for i, out in enumerate(output):
            print(f"Output[{i}] shape: {out.shape}")
    else:
        print(f"Output shape: {output.shape}")

    mlir_text = import_module(model, example_input)
    print(mlir_text)


if __name__ == "__main__":
    main(sys.argv[1])
