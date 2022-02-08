import mxnet as mx
import config

from tools import load_mxnet_model, preprocess_image, get_model_output, write_output


def main():
    ctx = mx.cpu()
    image = preprocess_image(config.image_name, config.input_size)
    model = load_mxnet_model(ctx, config.mxnet_model_prefix, config.epoch, image)
    model_out = get_model_output(model, image)
    write_output(config.mxnet_output_file, model_out)
    print(f'Done! Check {config.mxnet_output_file}')


if __name__ == "__main__":
    main()
