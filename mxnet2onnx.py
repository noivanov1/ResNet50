import argparse
import mxnet as mx
import onnx
import numpy as np
import config

from mxnet.contrib import onnx as onnx_mxnet
from tools import write_logfile
from prettytable import PrettyTable
from ast import literal_eval
from typing import Tuple


def conversion_mxnet2onnx(mxnet_prefix: str, onnx_model: str, input_shape: Tuple[int, int, int, int]) -> str:
    """
    Conversion MXNet to ONNX model.
    """
    sym = f"{mxnet_prefix}-symbol.json"
    params = f"{mxnet_prefix}-0000.params"
    return onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_model, verbose=True)


def create_log() -> str:
    """
    Log file containing MXNet and ONNX packages versions.
    """
    headers = [" ", "Framework", "Version"]
    log_table = PrettyTable(headers)
    log_table.add_row(["Input_model", "MXNet", mx.__version__])
    log_table.add_row(["Dist_model", "ONNX", onnx.__version__])
    return log_table.get_string()


parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default=config.mxnet_model_prefix, type=str, help="prefix of the origin MXNet model")
parser.add_argument("--dist_model", default=config.onnx_model_name, type=str, help="name of the output ONNX model")
parser.add_argument("--input_shape", default=config.input_shape, help="input shape of the origin MXNet model")
parser.add_argument("--log_file", default=config.mxnet2onnx_log, type=str, help="write conversion log file")
args = parser.parse_args()


def main():
    """
    Model conversion.
    """
    converted_model = conversion_mxnet2onnx(args.prefix, args.dist_model, literal_eval(str(args.input_shape)))
    log_txt = create_log()
    write_logfile(args.log_file, log_txt)
    print(f"Done! Check {converted_model} and {args.log_file}")


if __name__ == "__main__":
    main()
