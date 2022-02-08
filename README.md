# ResNet50
Project for conversion **MXNet** ResNet50 to **ONNX** framework. See [Troubleshooting](#Troubleshooting).

## Requirements
python 3.6 \
mxnet 1.6 \
onnx 1.6 \
onnxruntime 1.6 \
prettytable 2.5 \
pillow 8.4 

## Environment 
1. Create _virtual env_ and install packages via _pip install -r requirements.txt_
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


## Troubleshooting 
* _**ValidationError:** Unrecognized attribute: spatial for operator BatchNormalization._ \
**Solution (bad):** Downgrade onnx to 1.3.0 version \
**Solution (good):** Change code in MXNet package:
https://github.com/apache/incubator-mxnet/pull/18846/files
------------------------
* _**BroadcastIterator:**:Init(int64_t, int64_t) axis == 1 || axis == largest was false. Attempting to broadcast an axis by a dimension other than 1. 64 by 112_\
**Solution (good):** Change code in MXNet package:
https://github.com/apache/incubator-mxnet/commit/f1a6df82a40d1d9e8be6f7c3f9f4dcfe75948bd6