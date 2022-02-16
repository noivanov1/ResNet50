import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))

import config
import args_parser

from test_tools import read_embedding_file, dim_test, absolute_error, relative_error, create_log
from tools import write_logfile


def main():
    mxnet_embedding = read_embedding_file(config.mxnet_output_file)
    onnx_mxnet_embedding = read_embedding_file(config.onnx_mxnet_output_file)
    onnxruntime_embedding = read_embedding_file(config.onnxruntime_output_file)

    dim_test(mxnet_embedding, onnx_mxnet_embedding)
    dim_test(mxnet_embedding, onnxruntime_embedding)

    max_abs_onnx = absolute_error(onnx_mxnet_embedding, mxnet_embedding)
    max_abs_onnxruntime = absolute_error(onnxruntime_embedding, mxnet_embedding)

    max_rel_onnx = relative_error(mxnet_embedding, onnx_mxnet_embedding)
    max_rel_onnxruntime = relative_error(mxnet_embedding, onnxruntime_embedding)

    test_log_table = create_log(["ONNX (MXNet back) inference","ONNX Runtime inference"], [max_abs_onnx, max_abs_onnxruntime], [max_rel_onnx, max_rel_onnxruntime])
    write_logfile(args_parser.args.test_mx_onnx, test_log_table)

    print(f'Done! Check {args_parser.args.test_mx_onnx}')


if __name__ == "__main__":
    main()
