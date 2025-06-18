import logreg_vizu as viz
import pandas as pd
import numpy as np

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
    W = np.zeros((X.shape[1], 1))
    b = np.zeros((1, 1))
    History = {
        'Losses': [],
        'Weights': [],
        'Biases': []
    }

    for epoch in range(epochs):
        Z = model(X, W, b)
        A = sigmoid(Z)
        loss = log_loss(Y, A)
        dW, db = gradient(X, Y, A)
        W, b = update_parameters(W, b, dW, db, learning_rate)

        History['Losses'].append(loss)
        History['Weights'].append(W.copy())
        History['Biases'].append(b.copy())

    return W, b, History


def main():
    dataset = pd.read_csv('datasets/dataset_train.csv')
    filtered_dataset = dataset.dropna()
    courses_dataset = filtered_dataset.select_dtypes(include='float')
    houses_dataset = filtered_dataset['Hogwarts House']
    houses = houses_dataset.unique()

    Weights = {}
    Biases = {}
    Histories = {}

    X = normalize(courses_dataset.to_numpy())

    for house in houses:
        Y = (houses_dataset == house).astype(int).to_numpy().reshape(-1, 1)
        W, b, History = logistic_regression(X, Y, epochs=2000)
        Weights[house] = W
        Biases[house] = b
        Histories[house] = History

    probs = {}
    for house in houses:
        Z = model(X, Weights[house], Biases[house])
        probs[house] = sigmoid(Z).flatten()
    probs_matrix = np.column_stack([probs[house] for house in houses])
    pred_indices = np.argmax(probs_matrix, axis=1)
    pred_houses = np.array(houses)[pred_indices]
    true_houses = houses_dataset.to_numpy()
    accuracy = np.mean(pred_houses == true_houses)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    viz.train_visualization(Histories, houses, true_houses, pred_houses, accuracy)

if __name__ == "__main__":
	main()