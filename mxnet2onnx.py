import mxnet as mx
import onnx
import numpy as np
import args_parser

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


def main():
    """
    Model conversion.
    """
    converted_model = conversion_mxnet2onnx(args_parser.args.prefix, args_parser.args.onnx_model,
                                            literal_eval(str(args_parser.args.input_shape)))
    log_txt = create_log()
    write_logfile(args_parser.args.log_conversion, log_txt)
    print(f"Done! Check {converted_model} and {args_parser.args.log_conversion}")


if __name__ == "__main__":
    main()
