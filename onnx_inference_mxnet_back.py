import mxnet as mx
import config

from inference_tools import preprocess_image, load_model_onnx, get_model_output_onnx_mxnet, write_output


def main():
    ctx = mx.cpu()
    input_image = preprocess_image(config.image_name, config.input_size)
    loaded_model = load_model_onnx(ctx, config.onnx_model_name, config.input_size)
    model_out = get_model_output_onnx_mxnet(loaded_model, input_image)
    write_output(config.onnx_mxnet_output_file, model_out)
    print(f'Done! Check {config.onnx_mxnet_output_file}')


if __name__ == "__main__":
    main()
