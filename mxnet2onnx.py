import os
import mxnet as mx
import onnx
import numpy as np

from ast import literal_eval
from typing import Tuple
from mxnet.contrib import onnx as onnx_mxnet
from prettytable import PrettyTable


def conversion_mxnet2onnx(mxnet_prefix: str, onnx_name: str, input_shape: Tuple[int, int, int, int]) -> str:  # noqa
    """
    Conversion MXNet to ONNX model.
    """
    if not os.path.exists("model_onnx/"):
        os.makedirs("model_onnx/")

    mxnet_model = f"model_mxnet/{mxnet_prefix}"
    output_onnx_model = f"/model_onnx/{onnx_name}.onnx"
    sym = f"{mxnet_model}-symbol.json"
    params = f"{mxnet_model}-0000.params"

    return onnx_mxnet.export_model(sym, params, [input_shape], np.float32, output_onnx_model, verbose=True)


def write_logfile(dist_model: str):
    """
    Writing log file.
    """
    headers = [" ", "Framework", "Version"]
    log_table = PrettyTable(headers)
    log_table.add_row(["Input", "MXNet", mx.__version__])
    log_table.add_row(["Output", "ONNX", onnx.__version__])
    log_table_txt = log_table.get_string()

    with open("embedder_mxnet2onnx_log.txt", "w") as logfile:
        logfile.write(f"Converted model is {dist_model}\n")
        logfile.write(log_table_txt)


def main():
    """
    Model conversion.
    """
    converted_model = conversion_mxnet2onnx('model', 'converted', literal_eval('1,3,112,112'))  # noqa
    print(f"Converted model is {converted_model}")
    write_logfile(converted_model)


if __name__ == "__main__":
    main()
