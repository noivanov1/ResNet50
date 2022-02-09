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


### Conversion
Set parameters for running inferences via command line or in **_config.py_** as default values.

1. Run
```console
python3 mxnet2onnx.py --prefix model_mxnet/model 
--dist_model model_onnx/converted_model.onnx --input_shape 1,3,112,112
```


### Inference

#### MXNet
1. Run
```console
python3 mxnet_inference.py --prefix model_mxnet/model --epoch 0 --input_image photo.bmp --input_shape 112,112--model_output  mxnet_inference.txt
```

#### ONNX (MXNet back)
1. Run 
```console
python3 onnx_inference_mxnet_back.py --onnx_model model_onnx/converted_model.onnx --input_image photo.bmp --input_shape 112,112 --model_output  onnx_mxnet_inference.txt
```


#### ONNX Runtime
1. Run
```console
python3 onnxruntime_inference.py --onnx_model model_onnx/converted_model.onnx --input_image photo.bmp --input_shape 112,112 --model_output  onnxruntime_inference.txt 
```


## Conversion problems 
* _**ValidationError:** Unrecognized attribute: spatial for operator BatchNormalization._ \
* _**BroadcastIterator:**:Init(int64_t, int64_t) axis == 1 || axis == largest was false. Attempting to broadcast an axis by a dimension other than 1. 64 by 112_

Resolved by patching [Environment](#Environment) par. 2 More in
* https://github.com/apache/incubator-mxnet/pull/18846/files
* https://github.com/apache/incubator-mxnet/commit/f1a6df82a40d1d9e8be6f7c3f9f4dcfe75948bd6