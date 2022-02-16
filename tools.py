import mxnet as mx
import onnx
import onnxruntime
import torch
import imp
import numpy as np


from mxnet.contrib.onnx.onnx2mx.import_model import import_model
from PIL import Image
from collections import namedtuple
from typing import Tuple


def preprocess_image(image_name: str, resize_shape: Tuple[int, int]) -> np.ndarray:
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


def preprocess_torch_image(image_array: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Preprocessing the image for pytorch model
    """
    image_tensor = torch.from_numpy(image_array)
    return image_tensor.to(device)


def load_model_mxnet(ctx: mx.context.Context, model_prefix: str, epoch: int, image: np.ndarray) -> \
                                                                                      mx.module.module.Module:
    """
    Load MXNet model
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=[])
    arg_params["data"] = mx.nd.array(image)
    model.bind(for_training=False, data_shapes=[("data", arg_params["data"].shape)])
    model.set_params(arg_params, aux_params, allow_missing=True)
    return model


def load_model_onnx(ctx: mx.context.Context, onnx_model_name: str, input_size: Tuple[int, int]) -> mx.module.module.Module:
    """
    Load ONNX model via MXNet
    """
    sym, arg_params, aux_params = import_model(onnx_model_name)
    loaded_model = mx.mod.Module(symbol=sym, context=ctx, label_names=[])
    loaded_model.bind(for_training=False, data_shapes=[('data', (1, 3, input_size[0], input_size[1]))])
    loaded_model.set_params(arg_params, aux_params, allow_missing=True)
    return loaded_model


def load_model_onnxruntime(onnx_model_name: str) -> onnx.onnx_ONNX_REL_1_6_ml_pb2.ModelProto:
    """
    Load ONNX model
    """
    return onnx.load(onnx_model_name)


def load_model_pytorch(kit_model: str, pytorch_model: str, device: torch.device):
    """
    Load PyTorch model
    """
    MainModel = imp.load_source('MainModel', kit_model)
    model = torch.load(pytorch_model).to(device)
    model.eval()
    return model


def get_model_output_onnx_mxnet(model: mx.module.module.Module, image: np.ndarray) -> np.ndarray:
    """
    Predict embedding
    """
    Batch = namedtuple("Batch", ["data"])
    model.forward(Batch([mx.nd.array(image)]))
    return np.squeeze(model.get_outputs()[0].asnumpy())


def get_model_output_onnxruntime(loaded_model, input_image):
    """
    Return embedding via ONNX Runtime
    """
    content = loaded_model.SerializeToString()
    sess = onnxruntime.InferenceSession(content)
    feed1 = {sess.get_inputs()[0].name: input_image}
    return sess.run(None, feed1)[0][0]


def get_model_output_pytorch(model, image: torch.Tensor) -> np.ndarray:
    """
    Predict PyTorch embedding
    """
    with torch.no_grad():
        output = model(image)
    return output[0].numpy()


def save_output(file_name: str, model_out: np.ndarray):
    """
    Write embedding to .npy file
    """
    np.save(file_name, np.asarray(model_out))


def write_logfile(file_name: str, log_txt: str):
    """
    Write embedding to .txt file
    """
    with open(file_name, "w") as logfile:
        logfile.write(log_txt)
