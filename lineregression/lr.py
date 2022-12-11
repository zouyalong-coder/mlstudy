import numpy as np

class LinearRegression(object):
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=0) -> None:
        '''
            1. 对数据进行预处理
            2. 得到特征个数
            3. 初始化参数矩阵
            polynomial_degree: 多项式
            sinusoid_degree: 正弦
        '''
        (data_processed, features_mean, features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))
    
    def train(self, alpha, num_iteration=500):
        """
            alpha: 学习率
        """
        cost_history = self.gradient_descent(alpha, num_iteration)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iteration):
        cost_history = []
        for _ in range(num_iteration):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def cost_function(self, data, labels):
        """
        损失计算方法。
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(data, labels)
        # 损失衡量标准可以自己定
        cost = (1/2) * np.dot(delta.T, delta)/num_examples
        # 这里通过打印来确定去哪一个元素，需要进一步了解
        return cost[0][0]
    
    def gradient_step(self, alpha):
        """
            梯度下降更新参数方法。
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha*(1/num_examples)*np.dot(delta.T, self.data).T
        self.theta = theta

    @staticmethod
    def hypothesis(data, theta):
        prediction = np.dot(data, theta)
        return prediction

    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0] 
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0] 
        prediction = LinearRegression.hypothesis(data_processed, self.theta)
        return prediction