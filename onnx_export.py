""" ONNX export script

Export PyTorch models as ONNX graphs.

This export script originally started as an adaptation of code snippets found at
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

The default parameters work with PyTorch 1.6 and ONNX 1.7 and produce an optimal ONNX graph
for hosting in the ONNX runtime (see onnx_validate.py). To export an ONNX model compatible
with caffe2 (see caffe2_benchmark.py and caffe2_validate.py), the --keep-init and --aten-fallback
flags are currently required.

Older versions of PyTorch/ONNX (tested PyTorch 1.4, ONNX 1.5) do not need extra flags for
caffe2 compatibility, but they produce a model that isn't as fast running on ONNX runtime.

Most new release of PyTorch and ONNX cause some sort of breakage in the export / usage of ONNX models.
Please do your research and search ONNX and PyTorch issue tracker before asking me. Thanks.

Copyright 2020 Ross Wightman
"""
import argparse

import timm
from timm.utils.model import reparameterize_model

from typing import Optional, Tuple, List

import torch
from typing import Optional, Tuple, List

import torch
import onnxruntime
import onnx

import numpy as np

def onnx_forward(onnx_file, example_input):
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnx_file, sess_options)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: example_input.numpy()})
    output = output[0]
    return output


def onnx_export(
        model: torch.nn.Module,
        output_file: str,
        example_input: Optional[torch.Tensor] = None,
        training: bool = False,
        verbose: bool = False,
        check: bool = True,
        check_forward: bool = False,
        batch_size: int = 1,
        input_size: Tuple[int, int, int] = None,
        opset: Optional[int] = None,
        dynamic_size: bool = False,
        aten_fallback: bool = False,
        keep_initializers: Optional[bool] = None,
        use_dynamo: bool = False,
        input_names: List[str] = None,
        output_names: List[str] = None,
):
    import onnx

    if training:
        training_mode = torch.onnx.TrainingMode.TRAINING
        model.train()
    else:
        training_mode = torch.onnx.TrainingMode.EVAL
    
    model.eval()

    if example_input is None:
        if not input_size:
            assert hasattr(model, 'default_cfg')
            input_size = model.default_cfg.get('input_size')

        # laod image from file image.png
        import cv2
        import numpy as np
        img = cv2.imread('image.png')
        img = cv2.resize(img, (input_size[1], input_size[2]))
        img = img.transpose(2, 0, 1)

        example_input = torch.from_numpy(img).float() # (1,3,288,288)

        # we need to normalize it with the iamgenet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # be sure with the dimensions
        mean = np.expand_dims(mean, axis=1)
        mean = np.expand_dims(mean, axis=2)
        std = np.expand_dims(std, axis=1)
        std = np.expand_dims(std, axis=2)

        example_input = (example_input / 255.0 - mean) / std

        example_input = example_input.unsqueeze(0).float()



        #example_input = torch.randn((batch_size,) + input_size, requires_grad=False)

    # Run model once before export trace, sets padding for models with Conv2dSameExport. This means
    # that the padding for models with Conv2dSameExport (most models with tf_ prefix) is fixed for
    # the input img_size specified in this script.

    # Opset >= 11 should allow for dynamic padding, however I cannot get it to work due to
    # issues in the tracing of the dynamic padding or errors attempting to export the model after jit
    # scripting it (an approach that should work). Perhaps in a future PyTorch or ONNX versions...
    with torch.no_grad():
        original_out = model(example_input)

    input_names = input_names or ["pixel_values"]
    output_names = output_names or ["logits"]

    dynamic_axes = {'pixel_values': {0: 'batch'}, 'logits': {0: 'batch'}}
    if dynamic_size:
        dynamic_axes['pixel_values'][2] = 'height'
        dynamic_axes['pixel_values'][3] = 'width'

    if aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX

    if use_dynamo:
        export_options = torch.onnx.ExportOptions(dynamic_shapes=dynamic_size)
        export_output = torch.onnx.dynamo_export(
            model,
            example_input,
            export_options=export_options,
        )
        export_output.save(output_file)
        torch_out = None
    else:
        torch_out = torch.onnx.export(
            model,
            example_input,
            output_file,
            #training=training_mode,
            #export_params=True,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            #keep_initializers_as_inputs=keep_initializers,
            #dynamic_axes=dynamic_axes,
            opset_version=15,
            #operator_export_type=export_type
        )

    if True:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model, full_check=True)  # assuming throw on error
        if True:
            import numpy as np
            print(original_out.detach().numpy())
            onnx_out = onnx_forward(output_file, example_input)
            print(original_out.numpy())
            if torch_out is not None:
                np.testing.assert_almost_equal(torch_out.detach().numpy(), onnx_out, decimal=3)
                np.testing.assert_almost_equal(original_out.numpy(), torch_out.detach().numpy(), decimal=5)
            else:
                np.testing.assert_almost_equal(original_out.numpy(), onnx_out, decimal=3)





parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('output', metavar='ONNX_FILE',
                    help='output model filename')
parser.add_argument('--opset', type=int, default=None,
                    help='ONNX opset to use (default: 10)')
parser.add_argument('--keep-init', action='store_true', default=False,
                    help='Keep initializers as input. Needed for Caffe2 compatible export in newer PyTorch/ONNX.')
parser.add_argument('--aten-fallback', action='store_true', default=False,
                    help='Fallback to ATEN ops. Helps fix AdaptiveAvgPool issue with Caffe2 in newer PyTorch/ONNX.')
parser.add_argument('--dynamic-size', action='store_true', default=False,
                    help='Export model width dynamic width/height. Not recommended for "tf" models with SAME padding.')
parser.add_argument('--check-forward', action='store_true', default=False,
                    help='Do a full check of torch vs onnx forward after export.')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')
parser.add_argument('--training', default=False, action='store_true',
                    help='Export in training mode (default is eval)')
parser.add_argument('--verbose', default=False, action='store_true',
                    help='Extra stdout output')
parser.add_argument('--dynamo', default=False, action='store_true',
                    help='Use torch dynamo export.')

def main():
    args = parser.parse_args()

    args.pretrained = True
    if args.checkpoint:
        args.pretrained = False

    #print("==> Creating PyTorch {} model".format(args.model))
    # NOTE exportable=True flag disables autofn/jit scripted activations and uses Conv2dSameExport layers
    # for models using SAME padding
    model = timm.create_model(
        "hf_hub:CristianR8/vgg19-model",
        num_classes=6,
        in_chans=3,
        pretrained=args.pretrained,
        exportable=True,
    )

    if args.reparam:
        model = reparameterize_model(model)

    onnx_export(
        model,
        args.output,
        opset=13,
        input_size=(3, 288, 288),
        use_dynamo=False
    )


if __name__ == '__main__':
    main()