import args_parser
import mxnet as mx
import config

from tools import preprocess_image, load_model_onnx, get_model_output_onnx_mxnet, write_output
from ast import literal_eval


def main():
    ctx = mx.cpu()
    input_image = preprocess_image(args_parser.args.input_image, literal_eval(str(args_parser.args.input_shape)))
    loaded_model = load_model_onnx(ctx, args_parser.args.onnx_model, literal_eval(str(args_parser.args.input_shape)))
    model_out = get_model_output_onnx_mxnet(loaded_model, input_image)
    write_output(args_parser.args.onnx_mx_model_output, model_out)
    print(f'Done! Check {args_parser.args.onnx_mx_model_output}')


if __name__ == "__main__":
    main()
