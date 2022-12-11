import numpy as np
from .normalize import normalize
from .generate_sinusoids import generate_sinusoids


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    num_examples = len(data)
    # 预处理
    data_processed = np.copy(data)
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    if normalize_data:
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)
        data_processed = data_normalized
    # 特征变换 sinusoidal
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)
    # 特征变换 polynomial
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)
    # 加一列：1，对应 x0，做矩阵运算
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))
    return data_processed, features_mean, features_deviation
    
