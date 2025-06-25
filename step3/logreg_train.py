import logreg_core as lrc
import logreg_visu as lrv
import pandas as pd
import numpy as np
import argparse

BEST_COURSES = ['Astronomy', 'Herbology']
WORST_COURSES = ['Care of Magical Creatures', 'Arithmancy']

def parse_args():
    parser = argparse.ArgumentParser(description='Train a logistic regression model on Hogwarts houses dataset.')
    parser.add_argument('-e', '--epochs', type=int, default=2000, help='Number of training epochs (default: 2000)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='Learning rate for training (default: 0.01)')
    parser.add_argument('-d', '--dataset', type=str, default='datasets/dataset_train.csv', help='Path to the training dataset (default: datasets/dataset_train.csv)')
    parser.add_argument('-b', '--best', action='store_true', help='Use the two best courses for training instead of all courses')
    parser.add_argument('-w', '--worst', action='store_true', help='Use the two worst courses for training instead of all courses')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = pd.read_csv(args.dataset)
    filtered_dataset = dataset.dropna()
    if args.best:
        courses_dataset = filtered_dataset[BEST_COURSES]
    elif args.worst:
        courses_dataset = filtered_dataset[WORST_COURSES]
    else:
        courses_dataset = filtered_dataset.select_dtypes(include='float')
    houses_dataset = filtered_dataset['Hogwarts House']
    houses = houses_dataset.unique()

    courses_matrix = courses_dataset.to_numpy()

    mean, std = lrc.fit(courses_matrix)
    courses_matrix = lrc.normalize(courses_matrix, mean, std)

    weights = []
    biases = []
    histories = lrc.History()

    for house in houses:
        houses_matrix = (houses_dataset == house).astype(int).to_numpy().reshape(-1, 1)
        W, b, history = lrc.logistic_regression(courses_matrix, houses_matrix, learning_rate=args.learning_rate, epochs=args.epochs)
        weights.append(W)
        biases.append(b)
        histories.append(history.W, history.b, history.L)
    
    np.savez('logreg_parameters.npz', mean=mean, std=std, weights=weights, biases=biases, courses=courses_dataset.columns, houses=houses)
    
    lrv.TrainVisualization(weights, biases, histories, courses_dataset, houses_dataset, houses, args.epochs)
    if courses_dataset.shape[1] == 2:
        lrv.TrainVisualization2D(weights, biases, histories, courses_dataset, houses_dataset, houses, args.epochs)

if __name__ == "__main__":
	main()