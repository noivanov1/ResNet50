import onnxruntime
import onnx
import config
import numpy as np

from PIL import Image
from tools import preprocess_image, write_output


def preprocess_image(image_name: str, resize_shape: tuple) -> np.ndarray:
    """
    Preprocessing the image for model
    """
    image = Image.open(image_name)
    image = np.array(image)
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)  # change to (c, h,w) order
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image


def load_model(onnx_model_name: str) -> onnx.onnx_ONNX_REL_1_6_ml_pb2.ModelProto:
    """
    Load ONNX model
    """
    return onnx.load(onnx_model_name)


def get_model_output(loaded_model, input_image):
    """
    Return embedding via ONNX Runtime
    """
    content = loaded_model.SerializeToString()
    sess = onnxruntime.InferenceSession(content)
    feed1 = {sess.get_inputs()[0].name: input_image}
    return sess.run(None, feed1)[0][0]


def main():
    loaded_onnx_model = load_model(config.onnx_model_name)
    print(type(loaded_onnx_model))
    input_image = preprocess_image(config.image_name, config.input_size)
    model_out = get_model_output(loaded_onnx_model, input_image)
    write_output(config.onnxruntime_output_file, model_out)
    print(f'Done! Check {config.onnxruntime_output_file}')


if __name__ == "__main__":
    main()
