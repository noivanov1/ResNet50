# General
mxnet_model_prefix = 'model_mxnet/model'
image_name = 'photo.bmp'
input_shape = 1,3,112,112
onnx_model_name = 'model_onnx/converted_model.onnx'


# Conversion MXNet to ONNX
mxnet2onnx_log = 'result/mxnet2onnx_log.txt'


# MXNet inference
epoch = 0
mxnet_output_file = 'result/mxnet_inference.npy'


# ONNX (MXNet back) inference
onnx_mxnet_output_file = 'result/onnx_mxnet_inference.npy'


# ONNX Runtime inference
onnxruntime_output_file = 'result/onnxruntime_inference.npy'

# Test converted models
test_embeddings_log = 'result/emb_test_log.txt'
