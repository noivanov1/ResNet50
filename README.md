# ResNet50
Project for conversion **MXNet** ResNet50 to **ONNX** framework.

## Requirements
python 3.6
mxnet 1.6
onnx 1.6
onnxruntime 1.6
prettytable 2.5
pillow 8.4

## Conversion
1. Create _virtual env_ and install libs from [Requirements](#Requirements)
2. Put **MXNet model** in _model_mxnet/_
3. Specify **_config.py_** with model prefix, input_size, input_picture, dist_model
4. Run **_mxnet2onnx.py_**