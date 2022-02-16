import torch
import args_parser

from tools import preprocess_image, preprocess_torch_image, load_model_pytorch, get_model_output_pytorch, save_output
from ast import literal_eval


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = preprocess_torch_image(preprocess_image(args_parser.args.input_image,
                                                    literal_eval(str(args_parser.args.input_shape))), device)
    model = load_model_pytorch(args_parser.args.kit_model, args_parser.args.pytorch_model, device)
    model_out = get_model_output_pytorch(model, image)
    save_output(args_parser.args.pytorch_model_output, model_out)
    print(f'Done! Check {args_parser.args.pytorch_model_output}')


if __name__ == "__main__":
    main()