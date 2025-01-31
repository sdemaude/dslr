# display four pie charts side by side, each representing the distribution of main hand preference in a Hogwarts house.

import pandas as pd
import matplotlib.pyplot as plt

# Load data, keeping only relevant columns
df = pd.read_csv('datasets/dataset_train.csv', usecols=['Hogwarts House', 'Best Hand'])

# Define Hogwarts houses
houses = df['Hogwarts House'].unique()

# Set up a 2x2 grid for the subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Adjust figure size as needed
axes = axes.flatten()  # Flatten the 2D array to easily iterate

# Colors and labels
colors = ['MediumPurple', 'LightGoldenrodYellow']
labels = ['Left', 'Right']

# Loop through each house and create pie charts
for i, house in enumerate(houses):
    # Count occurrences of each hand preference
    counts = df[df['Hogwarts House'] == house]['Best Hand'].value_counts()

    # Ensure all labels exist in counts (to avoid missing categories)
    counts = counts.reindex(labels, fill_value=0)

    # Plot pie chart
    axes[i].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
    axes[i].set_title(house)

plt.tight_layout()
plt.show()