import argparse
import config


parser = argparse.ArgumentParser()
# General
parser.add_argument("--input_image", default=config.image_name, type=str, help="name of the input image")
parser.add_argument("--input_shape", default=[config.input_shape[2], config.input_shape[3]],
                    help="input shape of the model")

# MXNet to ONNX conversion
parser.add_argument("--conversion_log", default=config.mxnet2onnx_log, type=str, help="write conversion log file")
parser.add_argument("--onnx_model", default=config.onnx_model_name, type=str, help="name of converted ONNX model")

# MXNet model
parser.add_argument("--prefix", default=config.mxnet_model_prefix, type=str, help="prefix of the origin MXNet model")
parser.add_argument("--epoch", default=config.epoch, type=int, help="epoch of the pretrained MXNet model")
parser.add_argument("--mxnet_model_output", default=config.mxnet_output_file, type=str, help="write embedding to file")

# ONNX (MXNet back) and ONNX Runtime
parser.add_argument("--onnx_mx_model_output", default=config.onnx_mxnet_output_file, type=str,
                    help="write embedding to file")
parser.add_argument("--onnxruntime_model_output", default=config.onnxruntime_output_file, type=str,
                    help="write embedding to file")

# PyTorch model
parser.add_argument("--pytorch_model", default=config.pytorch_model, type=str, help="name of converted PyTorch model")
parser.add_argument("--kit_model", default=config.kit_model, type=str, help="name of kit_model")
parser.add_argument("--pytorch_model_output", default=config.pytorch_output_file, type=str,
                    help="write embedding to file")

# Test outputs
parser.add_argument("--test_log", default=config.test_embeddings_log, type=str, help="write test log file")

args = parser.parse_args()



