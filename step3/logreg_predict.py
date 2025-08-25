import logreg_core as lrc
import pandas as pd
import numpy as np
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Predict Hogwarts houses using a trained logistic regression model.')
    parser.add_argument('-d', '--dataset', type=str, default='datasets/dataset_test.csv', help='Path to the test dataset (default: datasets/dataset_test.csv)')
    parser.add_argument('-m', '--model', type=str, default='logreg_parameters.npz', help='Path to the trained model parameters (default: logreg_parameters.npz)')
    return parser.parse_args()


def main():
	args = parse_args()


	if not os.path.exists(args.model):
		print(f"Error: Model file '{args.model}' does not exist.")
		exit(1)
	model = np.load(args.model, allow_pickle=True)
	mean = model['mean']
	std = model['std']
	weights = model['weights']
	biases = model['biases']
	courses = model['courses']
	houses = model['houses']

	dataset = pd.read_csv(args.dataset)

	mean_dict = dict(zip(courses, mean))
	courses_dataset = dataset[courses].fillna(mean_dict)

	courses_matrix = lrc.normalize(courses_dataset.to_numpy(), mean, std)

	predicted_houses = houses[lrc.predict(courses_matrix, weights, biases)]

	predictions_df = pd.DataFrame({
		'Index': courses_dataset.index,
		'Hogwarts House': predicted_houses
	})
	predictions_df.to_csv('houses.csv', index=False)


if __name__ == "__main__":
    main()
