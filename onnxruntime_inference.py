import config

from inference_tools import preprocess_image, load_model_onnxruntime, get_model_output_onnxruntime, write_output


def main():
    loaded_onnx_model = load_model_onnxruntime(config.onnx_model_name)
    input_image = preprocess_image(config.image_name, config.input_size)
    model_out = get_model_output_onnxruntime(loaded_onnx_model, input_image)
    write_output(config.onnxruntime_output_file, model_out)
    print(f'Done! Check {config.onnxruntime_output_file}')


if __name__ == "__main__":
    main()
