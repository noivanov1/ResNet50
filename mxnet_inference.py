import mxnet as mx
import config

from inference_tools import load_model_mxnet, preprocess_image, get_model_output_onnx_mxnet, write_output


def main():
    ctx = mx.cpu()
    image = preprocess_image(config.image_name, config.input_size)
    model = load_model_mxnet(ctx, config.mxnet_model_prefix, config.epoch, image)
    model_out = get_model_output_onnx_mxnet(model, image)
    write_output(config.mxnet_output_file, model_out)
    print(f'Done! Check {config.mxnet_output_file}')


if __name__ == "__main__":
    main()
