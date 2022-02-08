import mxnet as mx
import numpy as np

from PIL import Image
from collections import namedtuple


def preprocess_image(image_name: str, resize_shape: tuple) -> np.ndarray:
    """
    Preprocessing the image for model
    """
    image = Image.open(image_name)
    image = image.resize((resize_shape[0], resize_shape[1]))
    image = np.array(image)
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)  # change to (c, h,w) order
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image


def load_mxnet_model(ctx: mx.context.Context, model_prefix: str, epoch: int, image: np.ndarray) -> \
                                                                                      mx.module.module.Module:
    """
    Load MXNet model
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers["fc1_output"]
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=[])
    arg_params["data"] = mx.nd.array(image)
    model.bind(for_training=False, data_shapes=[("data", arg_params["data"].shape)])
    model.set_params(arg_params, aux_params, allow_missing=True)
    return model


def get_model_output(model: mx.module.module.Module, image: np.ndarray) -> np.ndarray:
    """
    Predict embedding
    """
    Batch = namedtuple("Batch", ["data"])
    model.forward(Batch([mx.nd.array(image)]))
    return np.squeeze(model.get_outputs()[0].asnumpy())


def write_output(file_name: str, model_out: np.ndarray):
    """
    Write embedding to .txt file
    """
    with open(file_name, 'w') as out:
        for i in range(len(model_out)):
            out.write(str(model_out[i]) + '\n')