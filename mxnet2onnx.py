import os
import mxnet as mx
import onnx
import numpy as np
import config

from typing import Tuple
from mxnet.contrib import onnx as onnx_mxnet
from prettytable import PrettyTable


def conversion_mxnet2onnx(mxnet_prefix: str, onnx_model: str, input_shape: Tuple[int, int, int, int]) -> str:  # noqa
    """
    Conversion MXNet to ONNX model.
    """
    sym = f"{mxnet_prefix}-symbol.json"
    params = f"{mxnet_prefix}-0000.params"
    return onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_model, verbose=True)


def create_log(dist_model: str) -> str:
    """
    Log file containing MXNet and ONNX packages versions.
    """
    headers = [" ", "Framework", "Version"]
    log_table = PrettyTable(headers)
    log_table.add_row(["Input", "MXNet", mx.__version__])
    log_table.add_row(["Output", "ONNX", onnx.__version__])
    return log_table.get_string()


def write_logfile(file_name: str, log_txt: str):
    with open(file_name, "w") as logfile:
        logfile.write(log_txt)


def main():
    """
    Model conversion.
    """
    converted_model = conversion_mxnet2onnx(config.mxnet_model_prefix, config.onnx_model_name, config.conversion_input_size)  # noqa
    print(f"Converted model is {converted_model}")
    log_txt = create_log(converted_model)
    write_logfile(config.mxnet2onnx_log, log_txt)


if __name__ == "__main__":
    main()
