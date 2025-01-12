from transformers import AutoModel, AutoModelForImageClassification
from timm import create_model
import torch
import os

# Define model and ONNX paths
model_name = "CristianR8/resnet50-model"
onnx_path = "model.onnx"

model = AutoModel.from_pretrained("CristianR8/resnet50-model")  

# Dummy input for export
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(
    model.to('cpu'),
    dummy_input,
    onnx_path,
    input_names=["pixel_values"],
    output_names=["logits"],
)

print(f"Model successfully exported to {onnx_path}")
