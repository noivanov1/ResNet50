import numpy as np
import config

from prettytable import PrettyTable


class BreakException(Exception):
    pass


def dim_test(vector_1: np.ndarray, vector_2: np.ndarray, vector_3: np.ndarray):
    if len(vector_1) == len(vector_2) and len(vector_1) == len(vector_3):
        print('Embeddings size match')
    else:
        print('Embeddings size do not match')
        raise BreakException('Embeddings size do not match, check embeddings and conversion code')


def read_embedding_file(embedding_file: str) -> np.ndarray:
    embedding = []
    with open(embedding_file) as file:
        for element in file:
            embedding.append(float(element))
    return np.asarray(embedding)


def absolute_error(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    return np.max(abs(vector_1 - vector_2))


def relative_error(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    errors_list = []
    for i in range(len(vector_1)):
        errors_list.append(abs((vector_1[i] - vector_2[i]) / vector_1[i]))
    return np.max(errors_list)


def create_log(max_abs_onnx__mxnet: float, max_abs_onnxruntime__mxnet: float, mxnet_onnx: float,\
               mxnet_onnxruntime: float) -> str:
    """
    Log file containing MXNet and ONNX packages versions.
    """
    headers = ["MAX Errors to original MXNet model ", "Max Absolute error", "Max Relative error"]
    log_table = PrettyTable(headers)
    log_table.add_row(["ONNX (MXNet back) inference", max_abs_onnx__mxnet, mxnet_onnx])
    log_table.add_row(["ONNX Runtime inference", max_abs_onnxruntime__mxnet, mxnet_onnxruntime])
    return log_table.get_string()


def write_logfile(file_name: str, log_txt: str):
    """
    Write embedding to .txt file
    """
    with open(file_name, "w") as logfile:
        logfile.write(log_txt)


def main():
    mxnet_embedding = read_embedding_file(config.mxnet_output_file)
    onnx_mxnet_embedding = read_embedding_file(config.onnx_mxnet_output_file)
    onnxruntime_embedding = read_embedding_file(config.onnxruntime_output_file)
    dim_test(mxnet_embedding, onnx_mxnet_embedding, onnxruntime_embedding)
    max_abs_onnx__mxnet = absolute_error(onnx_mxnet_embedding, mxnet_embedding)
    max_abs_onnxruntime__mxnet = absolute_error(onnxruntime_embedding, mxnet_embedding)

    mxnet_onnx = relative_error(mxnet_embedding, onnx_mxnet_embedding)
    mxnet_onnxruntime = relative_error(mxnet_embedding, onnxruntime_embedding)
    create_log()


if __name__ == "__main__":
    main()
