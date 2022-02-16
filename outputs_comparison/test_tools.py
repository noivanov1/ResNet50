import numpy as np

from prettytable import PrettyTable


def dim_test(vector_1: np.ndarray, vector_2: np.ndarray):
    """
    Test MXNet, ONNX and ONNX Runtime embeddings size compatibility.
    """
    if len(vector_1) != len(vector_2):
        raise Exception('Embeddings size do not match, check embeddings and conversion code')


def read_embedding_file(embedding_file: str) -> np.ndarray:
    """
    Read embeddings from .npy file.
    """
    return np.load(embedding_file)


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


def create_log(model_outputs: list, max_abs: list, max_rel:list) -> str:
    """
    Log file containing table of ONNX (MXNet back) and ONNX Runtime embeddings errors.
    """
    headers = ["MAX Errors to original MXNet model ", "Max Absolute error", "Max Relative error"]
    log_table = PrettyTable(headers)
    for i in range(len(model_outputs)):
        log_table.add_row([model_outputs[i], max_abs[i], max_rel[i]])
    return log_table.get_string()
