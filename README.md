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
1. Create **_venv_** and install packages via 
```console
pip install --upgrade pip
pip install -r requirements.txt
```
2. Patch mxnet package (to resolve conversion problems) via
```console
patch patch_files/mx2onnx/_op_translations.py -i patch -o venv/lib/python3.6/site-packages/mxnet/contrib/onnx/mx2onnx/_op_translations.py
patch patch_files/onnx2mx/_op_translations.py -i patch -o venv/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py
```
3. Put **MXNet model** in _model_mxnet/_
4. Run
```console
python3 mxnet2onnx.py --prefix model_mxnet/model --dist_model model_onnx/converted_model.onnx --input_shape 1,3,112,112
```
specifying _--prefix_ and _--dist_model_ in command line or in **_config.py_**

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


## Conversion problems 
* _**ValidationError:** Unrecognized attribute: spatial for operator BatchNormalization._ \
* _**BroadcastIterator:**:Init(int64_t, int64_t) axis == 1 || axis == largest was false. Attempting to broadcast an axis by a dimension other than 1. 64 by 112_

Resolved by patching [Environment](#Environment) par. 2 More in
* https://github.com/apache/incubator-mxnet/pull/18846/files
* https://github.com/apache/incubator-mxnet/commit/f1a6df82a40d1d9e8be6f7c3f9f4dcfe75948bd6