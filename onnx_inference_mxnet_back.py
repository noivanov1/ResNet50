import numpy as np
import mxnet as mx
import config

from tools import preprocess_image
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
from collections import namedtuple


def load_model(ctx: mx.context.Context, onnx_model_name: str, input_size: tuple) -> mx.module.module.Module:
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


def get_model_output(loaded_model: mx.module.module.Module, input_picture: mx.ndarray.ndarray.NDArray) -> np.ndarray:
    """
    Predict embedding
    """
    Batch = namedtuple("Batch", ["data"])
    loaded_model.forward(Batch([mx.nd.array(input_picture)]))
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
    input_image = preprocess_image(config.image_name, config.input_size)
    loaded_model = load_model(ctx, config.onnx_model_name, config.input_size)
    model_out = get_model_output(loaded_model, input_image)
    write_output(config.onnx_mxnet_output_file, model_out)
    print(f'Done! Check {config.onnx_mxnet_output_file}')


if __name__ == "__main__":
    main()
