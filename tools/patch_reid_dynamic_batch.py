#!/usr/bin/env python3
"""Patch an ONNX ReID model for dynamic-batch inference.

Some torchreid OSNet exports contain a fixed `Reshape` constant `[1, -1]`
after global average pooling. With a batch of N images this flattens the batch
into the feature dimension (N*D), and the following Gemm fails with a
`K_a != K_b` error. Replacing the constant with `[0, -1]` and marking the input
batch dimension as dynamic makes the same weights work with batch > 1.

Usage:
    python patch_reid_dynamic_batch.py input.onnx output.onnx
"""

import argparse

import numpy as np
import onnx


def patch(input_path: str, output_path: str) -> None:
    model = onnx.load(input_path, load_external_data=False)
    graph = model.graph

    if len(graph.input) != 1:
        raise SystemExit(f"expected exactly one graph input, got {len(graph.input)}")

    batch_dim = graph.input[0].type.tensor_type.shape.dim[0]
    batch_dim.ClearField("dim_value")
    batch_dim.dim_param = "batch"

    changed = 0
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name != "value" or attribute.t.data_type != onnx.TensorProto.INT64:
                continue
            values = np.frombuffer(attribute.t.raw_data, dtype=np.int64)
            if values.size == 2 and values[0] == 1 and values[1] == -1:
                new_values = np.array([0, -1], dtype=np.int64)
                attribute.t.raw_data = new_values.tobytes()
                changed += 1

    if changed == 0:
        print("warning: no [1, -1] Reshape constant found; model may already support dynamic batch")

    onnx.save(model, output_path)
    print(f"saved {output_path} (patched constants: {changed})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="source ONNX model")
    parser.add_argument("output", help="destination ONNX model")
    args = parser.parse_args()
    patch(args.input, args.output)


if __name__ == "__main__":
    main()
