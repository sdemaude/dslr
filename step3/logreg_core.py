import numpy as np

class History:
    def __init__(self):
        self.W = []
        self.b = []
        self.L = []

    def append(self, W, b, L):
        self.W.append(W)
        self.b.append(b)
        self.L.append(L)

def normalize(data):
	mean = data.mean(axis=0)
	std = data.std(axis=0)
	return (data - mean) / std

def model(X, W, b):
	return X @ W + b

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

def log_loss(Y, A, epsilon=1e-15):
    m = Y.shape[0]
    A = np.clip(A, epsilon, 1 - epsilon)
    return -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

def gradient(X, Y, A):
    m = Y.shape[0]
    dZ = A - Y
    dW = 1/m * X.T @ dZ
    db = 1/m * np.sum(dZ)
    return dW, db

def update_parameters(W, b, dW, db, learning_rate):
	W = W - learning_rate * dW
	b = b - learning_rate * db
	return W, b

def logistic_regression(X, Y, learning_rate=0.01, epochs=1000):
    W = np.random.randn(X.shape[1], 1)
    b = np.zeros((1, 1))
    history = History()

    for epoch in range(epochs):
        Z = model(X, W, b)
        A = sigmoid(Z)
        L = log_loss(Y, A)
        history.append(W, b, L)
        dW, db = gradient(X, Y, A)
        W, b = update_parameters(W, b, dW, db, learning_rate)

    return W, b, history

def predict(X, weights, biases):
    W = np.array(weights)
    b = np.array(biases)
    probabilities = sigmoid(model(X, W, b))
    return np.argmax(probabilities, axis=0).flatten()