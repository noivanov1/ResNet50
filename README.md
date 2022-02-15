Project for conversion **MXNet** ResNet to **ONNX** framework.

| Models     | Conversion <br/>support |
|------------|-------------------------|
| ResNet-18  | &check;                 |
| ResNet-34  | &check;                 |
| ResNet-50  | &check;                 |
| ResNet-101 | &check;                 |



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
patch -u venv/lib/python3.6/site-packages/mxnet/contrib/onnx/mx2onnx/_op_translations.py -i patch_files/mx2onnx/_op_translations.patch
patch -u venv/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py -i patch_files/onnx2mx/_op_translations.patch
```
3. Download model via
```console
wget http://data.mxnet.io/models/imagenet/resnet/***-layers/resnet-***-0000.params
wget http://data.mxnet.io/models/imagenet/resnet/***-layers/resnet-***-symbol.json
```
where *** is 18, 34, 50 or 101.
4. Put **MXNet model** in _model_mxnet/_


### Conversion
Set parameters for running inferences via command line or in **_config.py_** as default values.

```console
python3 mxnet2onnx.py --prefix model_mxnet/model --dist_model model_onnx/converted_model.onnx --input_shape 1,3,112,112
```


### Inference

#### MXNet
```console
python3 mxnet_inference.py --prefix model_mxnet/model --epoch 0 --input_image photo.bmp --input_shape 112,112 --mxnet_model_output  mxnet_inference.txt
```

#### ONNX (MXNet back)
```console
python3 onnx_inference_mxnet_back.py --onnx_model model_onnx/converted_model.onnx --input_image photo.bmp --input_shape 112,112 --onnx_mx_model_output  onnx_mxnet_inference.txt
```


#### ONNX Runtime
```console
python3 onnxruntime_inference.py --onnx_model model_onnx/converted_model.onnx --input_image photo.bmp --input_shape 112,112 --onnxruntime_model_output  onnxruntime_inference.txt 
```


### Embeddings comparison
```console
python3 test_converted_outputs.py
```
#### ResNet-18
| MAX Errors to original <br/>MXNet model output | Max Absolute error | Max Relative error |
|------------------------------------------------|--------------------|--------------------|
| ONNX (MXNet back)                              | 3.1600000e-07      | 5.163130e-06       |
| ONNX Runtime                                   | 3.3999999e-07      | 8.084645e-06       |

#### ResNet-34
| MAX Errors to original <br/>MXNet model output | Max Absolute error | Max Relative error |
|------------------------------------------------|--------------------|--------------------|
| ONNX (MXNet back)                              | 7.1000000e-07      | 7.511862e-06       |
| ONNX Runtime                                   | 2.2100000e-06      | 1.190852e-05       |

#### ResNet-50
| MAX Errors to original <br/>MXNet model output | Max Absolute error | Max Relative error |
|------------------------------------------------|--------------------|--------------------|
| ONNX (MXNet back)                              | 2.456036e-06       | 0.092718           |
| ONNX Runtime                                   | 2.457186e-06       | 0.092712           |

#### ResNet-101
| MAX Errors to original <br/>MXNet model output | Max Absolute error | Max Relative error |
|------------------------------------------------|--------------------|--------------------|
| ONNX (MXNet back)                              | 3.0100000e-07      | 6.652124e-06       |
| ONNX Runtime                                   | 1.3100000e-06      | 1.874722e-05       |

## Conversion problems 
* _**ValidationError:** Unrecognized attribute: spatial for operator BatchNormalization._ 
* _**BroadcastIterator:**:Init(int64_t, int64_t) axis == 1 || axis == largest was false. Attempting to broadcast an axis by a dimension other than 1. 64 by 112_

Resolved by patching [Environment](#Environment) par. 2. \
More on
* https://github.com/apache/incubator-mxnet/pull/18846/files
* https://github.com/apache/incubator-mxnet/commit/f1a6df82a40d1d9e8be6f7c3f9f4dcfe75948bd6