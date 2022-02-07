import numpy as np
import mxnet as mx
import config

from mxnet.contrib.onnx.onnx2mx.import_model import import_model
from collections import namedtuple


def image_preprocessing(image_name: str) -> mx.ndarray.ndarray.NDArray:
    """
    Preprocessing the image for model
    """
    loaded_image = mx.image.imread(image_name)
    loaded_image = mx.image.imresize(loaded_image, config.image_resize_to[0], config.image_resize_to[1])
    loaded_image = loaded_image.transpose((2, 0, 1))
    loaded_image = loaded_image.expand_dims(axis=0)
    loaded_image = loaded_image.astype(dtype='float32')
    return loaded_image


def get_model(ctx: mx.context.Context, onnx_model_name: str, input_size: tuple) -> mx.module.module.Module:
    """
    Load ONNX model via MXNet
    """
    sym, arg_params, aux_params = import_model(onnx_model_name)
    all_layers = sym.get_internals()
    sym = all_layers["fc1_output"]
    loaded_model = mx.mod.Module(symbol=sym, context=ctx, label_names=[])
    loaded_model.bind(for_training=False, data_shapes=[('data', (1, 3, input_size[0], input_size[1]))])
    loaded_model.set_params(arg_params, aux_params, allow_missing=True)
    return loaded_model


def model_output(loaded_model: mx.module.module.Module, input_picture: mx.ndarray.ndarray.NDArray) -> np.ndarray:
    """
    Predict embedding
    """
    Batch = namedtuple("Batch", ["data"])
    loaded_model.forward(Batch([input_picture]))
    embedding = np.squeeze(loaded_model.get_outputs()[0].asnumpy())
    return embedding


def write_output(file_name: str, model_out: np.ndarray):
    """
    Write embedding to .txt file
    """
    with open(file_name, 'w') as out:
        for i in range(len(model_out)):
            out.write(str(model_out[i]) + '\n')


def main():
    ctx = mx.cpu()
    input_image = image_preprocessing(config.image_name)
    loaded_model = get_model(ctx, config.onnx_model_name, config.input_size)
    model_out = model_output(loaded_model, input_image)
    write_output(config.onnx_output_file, model_out)
    print('Done.')


if __name__ == "__main__":
    main()
