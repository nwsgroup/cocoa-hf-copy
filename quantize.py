import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load the exported ONNX model
onnx_path = "model.onnx"
loaded_model = onnx.load(onnx_path)

# Get operators from the ONNX model
def get_operators(onnx_model):
    ops = set()
    for node in onnx_model.graph.node:
        ops.add(node.op_type)
    return ops

op_types = get_operators(loaded_model)
weight_type = QuantType.QUInt8 if 'Conv' in op_types else QuantType.QInt8

# Quantize the model
quantized_model_path = "model_quantized.onnx"
quantize_dynamic(
    model_input=onnx_path,
    model_output=quantized_model_path,
    weight_type=weight_type,
    extra_options=dict(EnableSubgraph=True)
)

# Save quantization configuration (example for JSON output)
import json
quantize_config = {
    'per_model_config': {
        'model': {
            'op_types': list(op_types),
            'weight_type': str(weight_type),
        }
    }
}

with open('quantize_config.json', 'w') as fp:
    json.dump(quantize_config, fp, indent=4)