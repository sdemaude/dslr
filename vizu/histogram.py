# Which Hogwarts course has a homogeneous score distribution between all four houses?

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    data = pd.read_csv('datasets/dataset_train.csv')

    numeric_data = data.select_dtypes(include='float')
    numeric_data['Hogwarts House'] = data['Hogwarts House']

    mean_scores = numeric_data.groupby('Hogwarts House').mean()
    std_scores = mean_scores.std()

    course_colors = ['FireBrick', 'MidnightBlue', 'OliveDrab', 'DarkSlateGray', 'Magenta', 'DarkOrange',
                     'Sienna', 'Goldenrod', 'Indigo', 'Aquamarine', 'DarkKhaki', 'Plum', 'LightSteelBlue']

    bars = std_scores.plot(kind='bar', figsize=(12, 6),
                    title='Standard Deviation of House Mean Scores per Course',
                    ylabel='Standard Deviation of House Mean Scores', xlabel='Courses', color=course_colors)

    # Add text labels above the bars
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
                 bar.get_height() + 5,               # Y position (slightly above the bar)
                 f'{bar.get_height():.2f}',          # Display value with 2 decimal places
                 ha='center', va='bottom', fontsize=10, color='black')

    max_y = np.ceil(std_scores.max() / 50) * 50
    plt.yticks(np.arange(0, max_y + 1, 50))

    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
