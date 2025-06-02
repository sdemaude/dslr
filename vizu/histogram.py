from scipy.stats import wasserstein_distance
from itertools import combinations
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


courses_color = ['FireBrick', 'MidnightBlue', 'OliveDrab', 'DarkSlateGray', 'DarkMagenta', 'DarkOrange',
            'Sienna', 'Goldenrod', 'Indigo', 'Aquamarine', 'DarkKhaki', 'Plum', 'LightSteelBlue']

houses_colors = ['DarkRed', 'Gold', 'RoyalBlue', 'DarkGreen']

def wasserstein(houses_dataset, courses):
    wasserstein_score = pd.DataFrame(columns=courses)
    for (nameA, houseA), (nameB, houseB) in combinations(houses_dataset, 2):
        row = pd.Series(index=courses)
        for course in courses:
            dist = wasserstein_distance(houseA[course].dropna(), houseB[course].dropna())
            row[course] = dist
        wasserstein_score = pd.concat([wasserstein_score, row.to_frame().T], ignore_index=True)
    return wasserstein_score.mean()

def plot_course_histogram(houses_dataset, course):
    plt.figure(figsize=(8, 5))
    for (house, group), color in zip(houses_dataset, houses_colors):
        plt.hist(group[course].dropna(), bins=20, alpha=0.5, label=house, color=color)
    plt.title(f'Histogram for {course}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def plot_std(ax, houses_dataset, courses):
    ax.clear()

    std_scores = houses_dataset[courses].mean().std()

    ax.set(title='Standard Deviation of Mean Scores per Hogwarts House per Course',
           ylabel='Standard Deviation', xlabel='Courses', yscale='log')

    bars = ax.bar(courses, std_scores, color=courses_color[:len(courses)])

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', (bar.get_x() + bar.get_width() / 2, h * 1.1),
                    ha='center', va='bottom', fontsize=10, color='black')
    
    ax.tick_params(axis='x', rotation=60)
    plt.draw()
    return bars


def plot_wass(ax, houses_dataset, courses):
    ax.clear()

    wasserstein_score = wasserstein(houses_dataset, courses)

    ax.set(title='Mean Wasserstein Distance per Hogwarts House per Course',
           ylabel='Mean Wasserstein Distance', xlabel='Courses', yscale='log')

    bars = ax.bar(courses, wasserstein_score, color=courses_color[:len(courses)])

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', (bar.get_x() + bar.get_width() / 2, h * 1.1),
                    ha='center', va='bottom', fontsize=10, color='black')

    ax.tick_params(axis='x', rotation=60)
    plt.draw()
    return bars

def main():
    dataset = pd.read_csv('datasets/dataset_train.csv')
    courses = dataset.select_dtypes(include='float').columns
    houses_dataset = dataset.groupby('Hogwarts House')

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(top=0.825, bottom=0.25)

    bars = plot_std(ax, houses_dataset, courses)

    ax_std = plt.axes([0.3, 0.88, 0.15, 0.075])
    ax_wass = plt.axes([0.55, 0.88, 0.15, 0.075])
    btn_std = Button(ax_std, 'Std Dev')
    btn_wass = Button(ax_wass, 'Wasserstein')

    def update_std(event):
        nonlocal bars
        bars = plot_std(ax, houses_dataset, courses)

    def update_wass(event):
        nonlocal bars
        bars = plot_wass(ax, houses_dataset, courses)

    btn_std.on_clicked(update_std)
    btn_wass.on_clicked(update_wass)

    def onclick(event):
        if event.inaxes == ax:
            for i, bar in enumerate(bars):
                if bar.contains(event)[0]:
                    plot_course_histogram(houses_dataset, courses[i])
                    break

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

if __name__ == '__main__':
    main()
