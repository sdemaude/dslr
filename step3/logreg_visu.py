import logreg_core as lrc
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import math

houses_colors = {
	'Gryffindor': 'DarkRed',
	'Ravenclaw': 'RoyalBlue',
	'Slytherin': 'DarkGreen',
	'Hufflepuff': 'Gold'
}

class TrainVisualization:
	def __init__(self, weights, biases, histories, courses_dataset, houses_dataset, houses, epochs, step=5):
		self.weights = weights
		self.biases = biases
		self.histories = histories
		self.courses_dataset = courses_dataset
		self.houses_dataset = houses_dataset
		self.houses = houses
		self.epochs = epochs
		self.step = step
		self.frames = math.ceil(epochs / step)
		self.setup_plot()
		self.show()

	def setup_plot(self):
		self.course_matrix = lrc.normalize(self.courses_dataset.to_numpy())

		self.fig, [self.ax_loss, self.ax_confusion] = plt.subplots(1, 2)

		self.init_loss_plot()
		self.init_confusion_matrix()

		self.anim = animation.FuncAnimation(
			self.fig,
			lambda frame: self.update_plots(frame),
			frames=self.frames,
			interval=10,
			repeat=True
		)
		self.fig.canvas.mpl_disconnect(self.anim._first_draw_id)
		self.anim.is_running = False

		ax_anim_button = plt.axes([0.45, 0.01, 0.1, 0.075])
		self.anim_button = widgets.Button(ax_anim_button, 'Animate')

		self.anim_button.on_clicked(lambda _: self.toggle_animation())

	def show(self):
		plt.suptitle("Logistic Regression Training Visualization", fontsize=20)
		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.show()

	def compute_confusion_matrix(self, weights, biases):
		houses_predicted = self.houses[lrc.predict(self.course_matrix, weights, biases)]

		accuracy = np.mean(houses_predicted == self.houses_dataset)

		confusion_matrix = pd.crosstab(
			self.houses_dataset, houses_predicted,
			rownames=['Actual'], colnames=['Predicted'],
			dropna=False
		).reindex(index=self.houses, columns=self.houses, fill_value=0)

		return confusion_matrix, accuracy
	
	def preload_confusion_matrices(self):
		self.confusion_matrices = []
		for frame in range(self.frames - 1):
			epoch = frame * self.step
			weights = [house_weights[epoch] for house_weights in self.histories.W]
			biases = [house_biases[epoch] for house_biases in self.histories.b]
			confusion_matrix, accuracy = self.compute_confusion_matrix(weights, biases)
			self.confusion_matrices.append((confusion_matrix, accuracy))
		confusion_matrix, accuracy = self.compute_confusion_matrix(self.weights, self.biases)
		self.confusion_matrices.append((confusion_matrix, accuracy))

	def init_confusion_matrix(self):
		self.preload_confusion_matrices()
		confusion_matrix, accuracy = self.confusion_matrices[-1]

		self.plot_confusion = self.ax_confusion.matshow(confusion_matrix, cmap='Blues', alpha=0.7)
		self.ax_confusion.set_title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)', pad=20, fontsize=14)
		self.ax_confusion.set_xlabel('Predicted House')
		self.ax_confusion.set_ylabel('Actual House')
		ticks = np.arange(len(self.houses))
		self.ax_confusion.set_xticks(ticks)
		self.ax_confusion.set_yticks(ticks)
		self.ax_confusion.set_xticklabels(self.houses)
		self.ax_confusion.set_yticklabels(self.houses)
		self.ax_confusion.xaxis.tick_bottom()

		self.texts_confusion = []
		for (i, j), value in np.ndenumerate(confusion_matrix):
			text = self.ax_confusion.text(j, i, f'{value}', ha='center', va='center', color='black')
			self.texts_confusion.append(text)

	def update_confusion_matrix(self, frame, epoch):
		confusion_matrix, accuracy = self.confusion_matrices[frame]

		self.plot_confusion.set_data(confusion_matrix)

		self.ax_confusion.set_title(f'Confusion Matrix (Epoch {epoch}, Accuracy: {accuracy * 100:.2f}%)', pad=20, fontsize=14)

		for text, value in zip(self.texts_confusion, confusion_matrix.values.flatten()):
			text.set_text(f'{value}')

	def reset_confusion_matrix(self):
		confusion_matrix, accuracy = self.confusion_matrices[-1]

		self.plot_confusion.set_data(confusion_matrix)
		self.ax_confusion.set_title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)', pad=20, fontsize=14)

		for text, value in zip(self.texts_confusion, confusion_matrix.values.flatten()):
			text.set_text(f'{value}')
	
	def init_loss_plot(self):
		self.plots_loss = []
		for i, house in enumerate(self.houses):
			plot_loss, = self.ax_loss.plot(self.histories.L[i], label=house, color=houses_colors.get(house, 'gray'))
			self.plots_loss.append(plot_loss)

		self.ax_loss.set_title('Loss per epoch', pad=20, fontsize=14)
		self.ax_loss.set_xlabel('Epochs')
		self.ax_loss.set_ylabel('Loss')
		self.ax_loss.legend(title='Hogwarts Houses')
		self.ax_loss.margins(0.01)
		self.ax_loss.grid(True)
	
	def update_loss_plot(self, frame, epoch):
		for i, plot_loss in enumerate(self.plots_loss):
			x_data = np.arange(epoch + 1)
			y_data = self.histories.L[i][:epoch + 1]
			plot_loss.set_data(x_data, y_data)

	def reset_loss_plot(self):
		for i, plot_loss in enumerate(self.plots_loss):
			x_data = np.arange(self.epochs)
			y_data = self.histories.L[i]
			plot_loss.set_data(x_data, y_data)
	
	def update_plots(self, frame):
		epoch = min(frame * self.step, self.epochs)
		self.update_confusion_matrix(frame, epoch)
		self.update_loss_plot(frame, epoch)
	
	def reset_plots(self):
		self.anim.frame_seq = self.anim.new_frame_seq()
		self.reset_confusion_matrix()
		self.reset_loss_plot()
		self.fig.canvas.draw_idle()

	def toggle_animation(self):
		if self.anim.is_running:
			self.anim.pause()
			self.anim_button.label.set_text('Animate')
			self.reset_plots()
			self.anim.is_running = False
		else:
			if not self.anim._draw_was_started:
				self.anim._start()
			else:
				self.anim.resume()
			self.anim_button.label.set_text('Skip')
			self.anim.is_running = True



class TrainVisualization2D:
	def __init__(self, weights, biases, histories, courses_dataset, houses_dataset, houses, epochs, step=5):
		if len(courses_dataset.columns) != 2:
			raise ValueError("This visualization is designed for datasets with exactly two courses.")

		self.weights = weights
		self.biases = biases
		self.histories = histories
		self.courses_dataset = courses_dataset
		self.houses_dataset = houses_dataset
		self.houses = houses
		self.epochs = epochs
		self.step = step
		self.frames = math.ceil(epochs / step)
		self.setup_plot()
		self.show()

	def setup_plot(self):
		self.course_matrix = lrc.normalize(self.courses_dataset.to_numpy())

		self.fig, self.ax = plt.subplots()

		self.init_scatter_plot()

		"""
		self.anim = animation.FuncAnimation(
			self.fig,
			lambda frame: self.update_plots(frame),
			frames=self.frames,
			interval=10,
			repeat=True
		)
		self.fig.canvas.mpl_disconnect(self.anim._first_draw_id)
		self.anim.is_running = False

		ax_anim_button = plt.axes([0.45, 0.01, 0.1, 0.075])
		self.anim_button = widgets.Button(ax_anim_button, 'Animate')

		self.anim_button.on_clicked(lambda _: self.toggle_animation())
		"""

	def show(self):
		plt.suptitle("Logistic Regression Training Visualization", fontsize=20)
		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.show()
	
	def init_scatter_plot(self):
		houses_predicted = self.houses[lrc.predict(self.course_matrix, self.weights, self.biases)]
		#if houses_predicted == self.houses_dataset color is house colors else color is black
		self.norm_courses_by_house = lrc.normalize(self.courses_dataset).groupby(self.houses_dataset)
		self.scatter_plots = []
		for house, group in self.norm_courses_by_house:
			x = group.iloc[:, 0]
			y = group.iloc[:, 1]
			scatter = self.ax.scatter(x, y, s=50, alpha=0.7, label=house, color=houses_colors.get(house, 'gray'))
			self.scatter_plots.append(scatter)
		self.ax.legend(title='Hogwarts Houses')
		self.ax.set_xlabel(self.courses_dataset.columns[0])
		self.ax.set_ylabel(self.courses_dataset.columns[1])
		self.ax.set_title('Scatter Plot of Courses by House', pad=20, fontsize=14)