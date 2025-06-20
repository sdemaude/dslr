import pandas as pd
import matplotlib.pyplot as plt

houses_colors = {
	'Gryffindor': 'DarkRed',
	'Ravenclaw': 'RoyalBlue',
	'Slytherin': 'DarkGreen',
	'Hufflepuff': 'Gold'
}

dataset = pd.read_csv('datasets/dataset_train.csv')
houses_dataset = dataset.groupby('Hogwarts House')

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i, (house, group) in enumerate(houses_dataset):
    hand_counts = group['Best Hand'].value_counts()
    ordered_counts = hand_counts.reindex(['Left', 'Right'], fill_value=0)
    axes[i].pie(ordered_counts, labels=ordered_counts.index, autopct='%1.1f%%',
                colors=['lightcoral', 'lightblue'],
                explode=(0.1, 0),
                startangle=90,
                shadow={'color': houses_colors.get(house, 'gray')}
    )
    axes[i].set_title(f'{house} Best Hand Distribution')

plt.tight_layout()
plt.show()