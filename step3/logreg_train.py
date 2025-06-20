import logreg_core as lrc
import logreg_visu as lrv
import pandas as pd
import numpy as np


def main():
    dataset = pd.read_csv('datasets/dataset_train.csv')
    filtered_dataset = dataset.dropna()
    courses_dataset = filtered_dataset.select_dtypes(include='float')
    houses_dataset = filtered_dataset['Hogwarts House']
    houses = houses_dataset.unique()


    courses_matrix = lrc.normalize(courses_dataset.to_numpy())

    weights = []
    biases = []
    histories = lrc.History()

    for house in houses:
        houses_matrix = (houses_dataset == house).astype(int).to_numpy().reshape(-1, 1)
        W, b, history = lrc.logistic_regression(courses_matrix, houses_matrix, epochs=2000)
        weights.append(W)
        biases.append(b)
        histories.append(history.W, history.b, history.L)

    lrv.train_visualization(weights, biases, histories, courses_dataset, houses_dataset, houses)

if __name__ == "__main__":
	main()