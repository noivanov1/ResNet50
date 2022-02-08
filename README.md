# ResNet50
Project for conversion **MXNet** ResNet50 to **ONNX** framework.

## Requirements
python 3.6 \
mxnet 1.6 \
onnx 1.6 \
onnxruntime 1.6 \
prettytable 2.5 \
pillow 8.4 

## Environment 
1. Create _virtual env_ and install packages from [Requirements](#Requirements)
2. Put **MXNet model** in _model_mxnet/_

### Conversion
1. Specify **_config.py_** with _mxnet_model_prefix_, _conversion_input_size_, _onnx_model_name_, _log_file_
2. Run **_mxnet2onnx.py_**

### Inference
#### MXNet
1. Specify **_config.py_** with _mxnet_model_prefix_, _image_name_, _input_size_, _epoch_, _mxnet_output_file_
2. Run **_mxnet_inference.py_**

#### ONNX (MXNet back)
1. Specify **_config.py_** with _onnx_model_name_, _image_name_, _input_size_, _onnx_mxnet_output_file_
2. Run **_onnx_inference_mxnet_back.py_**

#### ONNX Runtime
1. Specify **_config.py_** with _onnx_model_name_, _image_name_, _input_size_, _onnx_mxnet_output_file_
2. Run **_onnxruntime_inference.py_**