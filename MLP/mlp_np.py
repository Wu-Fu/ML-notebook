import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # 初始化网络结构和参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_ih = np.random.randn(self.input_size, self.hidden_size)
        self.bias_h = np.zeros((1, self.hidden_size))
        self.weights_ho = np.random.randn(self.hidden_size, self.output_size)
        self.bias_o = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        # sigmoid激活函数
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # sigmoid函数的导数
        return x * (1 - x)

    def forward(self, inputs):
        # 前向传播
        self.hidden = self.sigmoid(np.dot(inputs, self.weights_ih) + self.bias_h)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_ho) + self.bias_o)
        return self.output

    def backward(self, inputs, targets):
        # 反向传播
        output_error = targets - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        hidden_error = np.dot(output_delta, self.weights_ho.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)

        # BP传播公式
        self.weights_ho += self.learning_rate * np.dot(self.hidden.T, output_delta)
        self.bias_o += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_ih += self.learning_rate * np.dot(inputs.T, hidden_delta)
        self.bias_h += self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, inputs, targets, num_epochs):
        # 训练模型
        for i in range(num_epochs):
            for j in range(len(inputs)):
                output = self.forward(inputs[j])
                self.backward(inputs[j], targets[j])

    def predict(self, inputs):
        # 预测
        return self.forward(inputs)
