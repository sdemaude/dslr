from scipy.stats import wasserstein_distance
from itertools import combinations
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('datasets/dataset_train.csv')
courses = dataset.select_dtypes(include='float').columns
houses_dataset = dataset.groupby('Hogwarts House')
colors = ['FireBrick', 'MidnightBlue', 'OliveDrab', 'DarkSlateGray', 'Magenta', 'DarkOrange',
            'Sienna', 'Goldenrod', 'Indigo', 'Aquamarine', 'DarkKhaki', 'Plum', 'LightSteelBlue']

def wasserstein(houses_dataset, courses):
    wasserstein_score = pd.DataFrame(columns=courses)
    for (nameA, houseA), (nameB, houseB) in combinations(houses_dataset, 2):
        row = pd.Series(index=courses)
        for course in courses:
            dist = wasserstein_distance(houseA[course].dropna(), houseB[course].dropna())
            row[course] = dist
        wasserstein_score = pd.concat([wasserstein_score, row.to_frame().T], ignore_index=True)
    return wasserstein_score.mean()

def plot_std(ax):
    ax.clear()

    std_scores = houses_dataset[courses].mean().std()

    ax.set(title='Standard Deviation of Mean Scores per Hogwarts House per Course',
           ylabel='Standard Deviation', xlabel='Courses', yscale='log')

    bars = ax.bar(courses, std_scores, color=colors[:len(courses)])

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', (bar.get_x() + bar.get_width() / 2, h * 1.1),
                    ha='center', va='bottom', fontsize=10, color='black')
    
    ax.tick_params(axis='x', rotation=60)
    plt.draw()


def plot_wass(ax):
    ax.clear()

    wasserstein_score = wasserstein(houses_dataset, courses)

    ax.set(title='Mean Wasserstein Distance per Hogwarts House per Course',
           ylabel='Mean Wasserstein Distance', xlabel='Courses', yscale='log')

    bars = ax.bar(courses, wasserstein_score, color=colors[:len(courses)])

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', (bar.get_x() + bar.get_width() / 2, h * 1.1),
                    ha='center', va='bottom', fontsize=10, color='black')

    ax.tick_params(axis='x', rotation=60)
    plt.draw()

def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(top=0.825, bottom=0.25)

    plot_std(ax)

    ax_std = plt.axes([0.3, 0.88, 0.15, 0.075])
    ax_wass = plt.axes([0.55, 0.88, 0.15, 0.075])
    btn_std = Button(ax_std, 'Std Dev')
    btn_wass = Button(ax_wass, 'Wasserstein')

    btn_std.on_clicked(lambda event: plot_std(ax))
    btn_wass.on_clicked(lambda event: plot_wass(ax))

    plt.show()

if __name__ == '__main__':
    main()