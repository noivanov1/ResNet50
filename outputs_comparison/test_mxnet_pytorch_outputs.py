import numpy as np
import config

from tools import write_logfile
from prettytable import PrettyTable


def read_embedding_file(embedding_file: str) -> np.ndarray:
    """
    Read embeddings from file.
    """
    embedding = []
    with open(embedding_file) as file:
        for element in file:
            embedding.append(float(element))
    return np.asarray(embedding)


def dim_test(vector_1: np.ndarray, vector_2: np.ndarray):
    """
    Test MXNet, ONNX and ONNX Runtime embeddings size compatibility.
    """
    if len(vector_1) == len(vector_2):
        print('Embeddings size match')
    else:
        raise Exception('Embeddings size do not match, check embeddings and conversion code')


def absolute_error(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    Calculate absolute errors.
    """
    return np.max(abs(vector_1 - vector_2))


def relative_error(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    Calculate relative errors.
    First vector is STRONGLY MXNet model embedding.
    """
    errors_list = []
    for i in range(len(vector_1)):
        errors_list.append(abs((vector_1[i] - vector_2[i]) / vector_1[i]))
    return np.max(errors_list)


def create_log(max_abs_pytorch: float, max_rel_pytorch: float) -> str:
    headers = ["MAX Errors to original MXNet model ", "Max Absolute error", "Max Relative error"]
    log_table = PrettyTable(headers)
    log_table.add_row(["PyTorch inference", f"{max_abs_pytorch:.9f}", f"{max_rel_pytorch:.9f}"])
    return log_table.get_string()


def main():
    mxnet_embedding = read_embedding_file(config.mxnet_output_file)
    pytorch_embedding = read_embedding_file(config.pytorch_output_file)

    dim_test(mxnet_embedding, pytorch_embedding)
    max_abs_pytorch = absolute_error(mxnet_embedding, pytorch_embedding)
    max_rel_pytorch = relative_error(mxnet_embedding, pytorch_embedding)

    test_log = create_log(max_abs_pytorch, max_rel_pytorch)
    write_logfile(config.test_mxnet_onnx_log, test_log)

    print(f'Done! Check {config.test_mxnet_onnx_log}')


if __name__ == "__main__":
    main()