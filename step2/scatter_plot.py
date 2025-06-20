import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

houses_colors = {
	'Gryffindor': 'DarkRed',
	'Ravenclaw': 'RoyalBlue',
	'Slytherin': 'DarkGreen',
	'Hufflepuff': 'Gold'
}

def scatter_plot(dataset, course_x, course_y):
	plt.figure(figsize=(10, 10))
	houses = dataset['Hogwarts House'].unique()
	for i, house in enumerate(houses):
		house_dataset = dataset['Hogwarts House'] == house
		plt.scatter(
			dataset.loc[house_dataset, course_x],
			dataset.loc[house_dataset, course_y],
			alpha=0.6,
			color=houses_colors.get(house, 'gray'),
			label=house
		)
	plt.xlabel(course_x)
	plt.ylabel(course_y)
	plt.title(f'Scatter Plot: {course_x} vs {course_y}')
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.legend(title='House')
	plt.show(block=False)

def on_click(event, dataset, courses):
	if event.inaxes is None:
		return
	x, y = int(round(event.xdata)), int(round(event.ydata))
	if 0 <= x < len(courses) and 0 <= y < len(courses) and x != y:
		scatter_plot(dataset, courses[x], courses[y])

def main():
	dataset = pd.read_csv('datasets/dataset_train.csv')
	courses = dataset.select_dtypes(include='float').columns

	correlation_matrix = dataset[courses].corr().abs()

	# Set the diagonal values to NaN and mask them
	for i in range(len(courses)):
		correlation_matrix.iloc[i, i] = np.nan
	masked_correlation_matrix = np.ma.masked_invalid(correlation_matrix)
	
	# Create a colormap and set NaN values to grey
	cmap = plt.cm.get_cmap("coolwarm").copy()
	cmap.set_bad(color='SlateGray')
	
	# Create a scatter plot with the correlation matrix
	fig = plt.figure(figsize=(10, 8))
	im = plt.imshow(masked_correlation_matrix, cmap=cmap)
	plt.colorbar(label='Correlation Coefficient')
	plt.xticks(np.arange(len(courses)), courses, rotation=45, ha='right')
	plt.yticks(np.arange(len(courses)), courses)
	plt.title('Correlation Matrix of Hogwarts Courses')

	# Afficher la valeur dans chaque case
	for i in range(len(courses)):
		for j in range(len(courses)):
			value = correlation_matrix.iloc[i, j]
			if np.isnan(value):
				value = np.delete(correlation_matrix.iloc[i], j).mean()
			plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black', fontsize=8)

	fig.canvas.mpl_connect(
		'button_press_event',
		lambda event: on_click(event, dataset, courses)
	)

	plt.show()

if __name__ == '__main__':
	main()