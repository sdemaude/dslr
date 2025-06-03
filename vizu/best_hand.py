import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/dataset_train.csv', usecols=['Hogwarts House', 'Best Hand'])
houses = df['Hogwarts House'].unique()
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
colors = ['MediumPurple', 'LightGoldenrodYellow']
labels = ['Left', 'Right']

for i, house in enumerate(houses):
    counts = df[df['Hogwarts House'] == house]['Best Hand'].value_counts()
    counts = counts.reindex(labels, fill_value=0)

    axes[i].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
    axes[i].set_title(house)

plt.tight_layout()
plt.show()