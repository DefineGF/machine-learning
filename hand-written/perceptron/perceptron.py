from functools import reduce
class Perceptron(object):
    """
        input: 输入参数个数；
        activator: 激活函数
    """

    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]  # 权重向量初始化为0
        self.bias = 0.0  # 偏置项初始化为0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    # x_0 * w_0 + x_1 * w_1 + ... + bias -> func_activator()
    def predict(self, input_vec):
        weights = self.weights
        _weight = map(lambda x, w : x * w, input_vec, weights)
        _reduce = reduce(lambda a, b : a + b, list(_weight), 0.0)
        return self.activator(_reduce + self.bias)

    """
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
    """
    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)  # 把输入 & 目标值 打包在一起，成为样本的列表[(input_vec, label), ...]
        for (input_vec, label) in samples:
            output = self.predict(input_vec)  # 计算感知器在当前权重下的输出
            self._update_weights(input_vec, output, label, rate)  # 更新权重 & 偏置项

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        _map = map(lambda x, w : w + rate * delta * x, input_vec, self.weights)
        self.weights = list(_map)
        self.bias += rate * delta  # 更新偏置项
