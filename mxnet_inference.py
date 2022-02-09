import argparse
import mxnet as mx
import config

from tools import load_model_mxnet, preprocess_image, get_model_output_onnx_mxnet, write_output
from ast import literal_eval


parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default=config.mxnet_model_prefix, type=str, help="prefix of the origin MXNet model")
parser.add_argument("--epoch", default=config.epoch, type=int, help="epoch of the pretrained MXNet model")
parser.add_argument("--input_image", default=config.image_name, type=str, help="name of the input image")
parser.add_argument("--input_shape", default=[config.input_shape[2], config.input_shape[3]],
                    help="input shape of the model")
parser.add_argument("--model_output", default=config.mxnet_output_file, type=str, help="write embedding to file")
args = parser.parse_args()


def main():
    ctx = mx.cpu()
    image = preprocess_image(args.input_image, literal_eval(str(args.input_shape)))
    model = load_model_mxnet(ctx, args.prefix, args.epoch, image)
    model_out = get_model_output_onnx_mxnet(model, image)
    write_output(args.model_output, model_out)
    print(f'Done! Check {config.mxnet_output_file}')


if __name__ == "__main__":
    main()
