import onnxruntime
import onnx
import numpy as np
import config

from PIL import Image


def image_preprocessing(image_name: str) -> np.ndarray:
    """
    Preprocessing the image for model
    """
    image = Image.open(image_name)
    img = np.array(image)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to (c, h,w) order
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


def get_model(onnx_model_name: str):
    """
    Load ONNX model
    """
    return onnx.load(onnx_model_name)


def model_output(loaded_model, input_image):
    """
    Return embedding via ONNX Runtime
    """
    content = loaded_model.SerializeToString()
    sess = onnxruntime.InferenceSession(content)
    feed1 = {sess.get_inputs()[0].name: input_image}
    return sess.run(None, feed1)[0][0]


def write_output(file_name: str, model_out: np.ndarray):
    """
    Write embedding to .txt file
    """
    with open(file_name, 'w') as out:
        for i in range(len(model_out)):
            out.write(str(model_out[i]) + '\n')


def main():
    loaded_onnx_model = get_model(config.onnx_model_name)
    input_image = image_preprocessing(config.image_name)
    model_out = model_output(loaded_onnx_model, input_image)
    write_output(config.onnxruntime_output_file, model_out)
    print(f'Done! Check {config.onnxruntime_output_file}')


if __name__ == "__main__":
    main()
