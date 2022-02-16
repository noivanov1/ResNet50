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
    pytorch_embedding = read_embedding_file(config.pytorch_output_file)

    dim_test(mxnet_embedding, pytorch_embedding)
    max_abs_pytorch = absolute_error(mxnet_embedding, pytorch_embedding)
    max_rel_pytorch = relative_error(mxnet_embedding, pytorch_embedding)

    test_log = create_log(["PyTorch inference"], [max_abs_pytorch], [max_rel_pytorch])
    write_logfile(args_parser.args.test_mx_pytorch, test_log)

    print(f'Done! Check {args_parser.args.test_mx_pytorch}')


if __name__ == "__main__":
    main()