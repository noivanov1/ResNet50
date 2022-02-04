import sys
import numpy as np
import mxnet as mx
import os
from PIL import Image
from mxnet.contrib.onnx.onnx2mx.import_model import import_model


def get_model(ctx, model):
    image_size = (112, 112)
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    # Define and binds parameters to the network
    all_layers = sym.get_internals()
    sym = all_layers["fc1_output"]
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=[])
    model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params, allow_missing=True)
    return model


def get_feature(model, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = np.squeeze(model.get_outputs()[0].asnumpy())
    return embedding


ctx = mx.cpu()
# Path to ONNX model
model_name = '../model/embedder_1.6.onnx'
# Load ONNX model
model = get_model(ctx, model_name)

img_txt = 'onnx_1.6.txt'
image = 'photo_1.bmp'

image = Image.open(image)
img = np.array(image)

img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 1, 2)  # change to (c, h,w) order
# img = img[np.newaxis, :]  # extend to (n, c, h, w)
out = get_feature(model, img)
# print(img)
print(img.shape)
print(out)

open(img_txt, 'w').close()
with open(img_txt, "a") as file:
    for i in range(len(out)):
        file.write(str(out[i]) + '\n')