#What are the two features that are similar ?

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('datasets/dataset_train.csv')
courses = dataset.select_dtypes(include='float').columns
colors = ['FireBrick', 'MidnightBlue', 'OliveDrab', 'DarkSlateGray', 'Magenta', 'DarkOrange',
            'Sienna', 'Goldenrod', 'Indigo', 'Aquamarine', 'DarkKhaki', 'Plum', 'LightSteelBlue']

def main():
	correlation_matrix = dataset[courses].corr().abs()

	# Create a scatter plot with the correlation matrix
	plt.figure(figsize=(10, 8))
	im = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
	plt.colorbar(label='Correlation Coefficient')
	plt.xticks(np.arange(len(courses)), courses, rotation=45, ha='right')
	plt.yticks(np.arange(len(courses)), courses)
	plt.title('Correlation Matrix of Hogwarts Courses')

	# Afficher la valeur dans chaque case
	for i in range(len(courses)):
		for j in range(len(courses)):
			value = correlation_matrix.iloc[i, j]
			plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black', fontsize=8)

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()