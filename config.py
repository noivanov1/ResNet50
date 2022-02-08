# General
mxnet_model_prefix = 'model_mxnet/model'
image_name = 'photo.bmp'
input_size = (112, 112)
onnx_model_name = 'model_onnx/converted_model.onnx'


# Conversion MXNet to ONNX
mxnet2onnx_log = 'mxnet2onnx_log.txt'
conversion_input_size = [1, 3, 112, 112]


# MXNet inference
epoch = 0
mxnet_output_file = 'mxnet_inference.txt'


# ONNX (MXNet back) inference
onnx_mxnet_output_file = 'onnx_mxnet_inference.txt'


# ONNX Runtime inference
onnxruntime_output_file = 'onnxruntime_inference.txt'
