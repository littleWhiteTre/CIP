import numpy as np
import copy

N = 10
D_in = 100
H = 150
D_out = 10
lr = 1e-4

class BPNN:
    def __init__(self, N, D_in, H, D_out):
        self.N = N
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.w1 = np.random.randn(D_in, N)
        self.w2 = np.random.randn(N, D_out)

    def forward(self, x):
        self.x = x
        self.h = x.dot(self.w1)
        return self.h.dot(self.w2)

    def backward(self):
        self.grad_w2 = self.h.T.dot(self.grad_pred)
        self.grad_h = self.grad_pred.dot(self.w2.T)
        self.grad_w1 = self.x.T.dot(self.grad_h)

    def step(self, learning_rate):
        self.w1 = self.w1 - learning_rate * self.grad_w1
        self.w2 = self.w2 - learning_rate * self.grad_w2

    def MSE_loss(self, pred, y):
        self.grad_pred = 2 * (pred - y)
        return np.square(pred - y).sum()

    def cross_entropy_loss(self, pred, y):
        pred = self._softmax(pred)
        y_ = np.zeros(pred.shape, dtype=np.int)
        for i in range(y_.shape[0]):
            y_[i, y[i]] = 1
        sum = 0.
        for i in range(y_.shape[0]):
            rows = 0.
            for j in range(y_.shape[1]):
                rows = rows + y_[i, j]*np.log(pred[i, j])
            sum = sum + rows
        self.grad_pred = pred - 1.
        return - sum / y_.shape[0]
        
    def _softmax(self, inp):
        inp = copy.copy(inp)
        for i in range(inp.shape[0]):
            inp[i] = np.exp(inp[i])
            inp[i] = inp[i] / np.sum(inp[i])
        return inp


if __name__ == "__main__":
    x = np.random.randn(N, D_in)
    y = np.array(range(0, 10))
    model = BPNN(N, D_in, H, D_out)
    for epoch in range(100):
        pred = model.forward(x)
        loss = model.cross_entropy_loss(pred, y)
        model.backward()
        model.step(lr)
        # print(loss)
    pred = model.forward(x)
    print(pred)
    print('*'*40)
    print(model._softmax(pred))
    print('*'*40)
    print(y)