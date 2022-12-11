import numpy as np

from .normalize import normalize

def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """
        使用多项式方式丰富特征
    """
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]
    