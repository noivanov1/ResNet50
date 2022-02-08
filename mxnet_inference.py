import numpy as np
import mxnet as mx
import config

from collections import namedtuple
from tools import load_mxnet_model, preprocess_image


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


def main():
    ctx = mx.cpu()
    image = preprocess_image(config.image_name, config.input_size)
    model = load_mxnet_model(ctx, config.mxnet_model_prefix, config.epoch, image)
    model_out = get_model_output(model, image)
    write_output(config.mxnet_output_file, model_out)
    print(f'Done! Check {config.mxnet_output_file}')


if __name__ == "__main__":
    main()
