import args_parser

from tools import preprocess_image, load_model_onnxruntime, get_model_output_onnxruntime, save_output
from ast import literal_eval


def main():
    loaded_onnx_model = load_model_onnxruntime(args_parser.args.onnx_model)
    input_image = preprocess_image(args_parser.args.input_image, literal_eval(str(args_parser.args.input_shape)))
    model_out = get_model_output_onnxruntime(loaded_onnx_model, input_image)
    save_output(args_parser.args.onnxruntime_model_output, model_out)
    print(f'Done! Check {args_parser.args.onnxruntime_model_output}')


if __name__ == "__main__":
    main()
