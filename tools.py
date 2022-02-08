import mxnet as mx
import numpy as np

from PIL import Image


def preprocess_image(image_name: str, resize_shape: tuple) -> np.ndarray:
    """
    Preprocessing the image for model
    """
    image = Image.open(image_name)
    image = image.resize((resize_shape[0], resize_shape[1]))
    image = np.array(image)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
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