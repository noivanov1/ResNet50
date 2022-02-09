import argparse
import config

from inference_tools import preprocess_image, load_model_onnxruntime, get_model_output_onnxruntime, write_output
from ast import literal_eval


parser = argparse.ArgumentParser()
parser.add_argument("--onnx_model", default=config.onnx_model_name, type=str, help="name of converted ONNX model")
parser.add_argument("--input_image", default=config.image_name, type=str, help="name of the input image")
parser.add_argument("--input_shape", default=[config.input_shape[2], config.input_shape[3]],
                    help="input shape of the model")
parser.add_argument("--model_output", default=config.onnxruntime_output_file, type=str, help="write embedding to file")
args = parser.parse_args()


def main():
    loaded_onnx_model = load_model_onnxruntime(args.onnx_model)
    input_image = preprocess_image(args.input_image, literal_eval(str(args.input_shape)))
    model_out = get_model_output_onnxruntime(loaded_onnx_model, input_image)
    write_output(args.model_output, model_out)
    print(f'Done! Check {config.onnxruntime_output_file}')


if __name__ == "__main__":
    main()
