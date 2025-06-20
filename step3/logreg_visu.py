import logreg_core as lrc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

houses_colors = {
	'Gryffindor': 'DarkRed',
	'Ravenclaw': 'RoyalBlue',
	'Slytherin': 'DarkGreen',
	'Hufflepuff': 'Gold'
}

def train_visualization(weights, biases, histories, courses_dataset, houses_dataset, houses):

	course_matrix = lrc.normalize(courses_dataset.to_numpy())

	houses_predicted = houses[lrc.predict(course_matrix, weights, biases)]

	accuracy = np.mean(houses_predicted == houses_dataset)

	confusion_matrix = pd.crosstab(houses_dataset, houses_predicted, rownames=['Actual'], colnames=['Predicted'])

	fig, [ax_loss, ax_confusion] = plt.subplots(1, 2)

	for i, house in enumerate(houses):
		ax_loss.plot(histories.L[i], label=house, color=houses_colors.get(house, 'gray'))

	ax_loss.set_title('Loss History', pad=20, fontsize=14)
	ax_loss.set_xlabel('Epochs')
	ax_loss.set_ylabel('Loss')
	ax_loss.legend(title='Hogwarts Houses')
	ax_loss.margins(0)
	ax_loss.grid(True)


	ax_confusion.matshow(confusion_matrix, cmap='Blues', alpha=0.7)
	ax_confusion.set_title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)', pad=20, fontsize=14)
	ax_confusion.set_xlabel('Predicted House')
	ax_confusion.set_ylabel('Actual House')
	ticks = np.arange(len(houses))
	ax_confusion.set_xticks(ticks)
	ax_confusion.set_yticks(ticks)
	ax_confusion.set_xticklabels(houses)
	ax_confusion.set_yticklabels(houses)
	ax_confusion.xaxis.tick_bottom()

	for (i, j), value in np.ndenumerate(confusion_matrix):
		ax_confusion.text(j, i, f'{value}', ha='center', va='center', color='black')
	
	plt.suptitle("Logistic Regression Training Visualization", fontsize=20)
	plt.subplots_adjust(wspace=0.3, hspace=0.3)
	plt.show()

