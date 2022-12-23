import numpy as np


def normalize(features):
    '''
        归一化：(x-期望)/标准差
    '''
    features_normalized = np.copy(features).astype(float)
    # 计算均值
    features_mean = np.mean(features, 0)
    # 计算标准差
    features_deviation = np.std(features, 0)
    # 标准化操作：使取值中心集中到原点
    if features.shape[0] > 1:
        features_normalized -= features_mean
    # 防止除零
    features_deviation[features_deviation==0] = 1
    # 经此操作，取值范围为[-1, 1]，避免某个特征取值范围过大导致影响力过大
    features_normalized /= features_deviation
    return features_normalized, features_mean, features_deviation