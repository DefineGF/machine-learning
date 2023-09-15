from functools import reduce
import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(x))


class Node(object):

    def __init__(self, layer_index, node_index):
        """
        :param layer_index: 节点所属层编号
        :param node_index: 节点编号
        """
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []             # 上游连接 connection 列表
        self.upstream = []               # 下游连接 connection 列表
        self.output = 0
        self.delta = 0

    def set_output(self, output):                 # 设置节点的输出值。如果节点属于输入层会用到这个函数
        self.output = output

    def append_downstream_connection(self, conn): # 添加一个到下游节点的连接
        self.downstream.append(conn)

    def append_upstream_connection(self, conn): # 添加一个到上游节点的连接
        self.upstream.append(conn)

    def calc_output(self): # 计算节点输出
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self): # 计算 隐藏层 误差项 a_j * (1 - a_j) * sum(下游误差项 * 连接权值)
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label): # 计算 输出层 误差项 y_j * (1 - y_j) * (t_j - y_j)
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self): # 打印节点信息
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str
