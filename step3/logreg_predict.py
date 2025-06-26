#It takes as a parameter dataset_test.csv and a file containing the weights trained by previous program.
#In order to evaluate the performance of your classifier this second program will have to generate a prediction file houses.csv formatted exactly as follows:
# 
# Index,Hogwarts House
# 0,Gryffindor
# 1,Hufflepuff
# 2,Ravenclaw
# 3,Hufflepuff
# 4,Slytherin
# 5,Ravenclaw
# 6,Hufflepuff

#generate a file with the predictions of the logistic regression model using the weights trained by previous program

import logreg_core as lrc
import pandas as pd
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Predict Hogwarts houses using a trained logistic regression model.')
    parser.add_argument('-d', '--dataset', type=str, default='datasets/dataset_test.csv', help='Path to the test dataset (default: datasets/dataset_test.csv)')
    parser.add_argument('-m', '--model', type=str, default='logreg_parameters.npz', help='Path to the trained model parameters (default: logreg_parameters.npz)')
    return parser.parse_args()


def main():
	args = parse_args()

	model = np.load(args.model, allow_pickle=True)
	mean = model['mean']
	std = model['std']
	weights = model['weights']
	biases = model['biases']
	courses = model['courses']
	houses = model['houses']

	dataset = pd.read_csv(args.dataset)
	courses_dataset = dataset[courses].dropna()

	courses_matrix = lrc.normalize(courses_dataset.to_numpy(), mean, std)

	predicted_houses = houses[lrc.predict(courses_matrix, weights, biases)]

	predictions_df = pd.DataFrame({
		'Index': courses_dataset.index,
		'Hogwarts House': predicted_houses
	})
	predictions_df.to_csv('houses.csv', index=False)


if __name__ == "__main__":
    main()