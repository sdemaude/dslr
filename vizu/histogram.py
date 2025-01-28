# Which Hogwarts course has a homogeneous score distribution between all four houses?

import matplotlib.pyplot as plt
import pandas as pd

def main():
    data = pd.read_csv('datasets/dataset_train.csv')
    numeric_data = data.select_dtypes(include='float')
    numeric_data['Hogwarts House'] = data['Hogwarts House']

    mean_scores = numeric_data.groupby('Hogwarts House').mean()
    std_scores = mean_scores.std(axis=1)

    house_colors = ['darkred', 'gold', 'midnightblue', 'darkgreen']
    std_scores.plot(kind='bar', 
                    title='Standard deviation of the scores for each house', 
                    ylabel='Standard Deviation of the Scores', 
                    xlabel='Hogwarts House', 
                    color=house_colors)

    plt.ylim(4000, std_scores.max() + 1)
    plt.show()

if __name__ == '__main__':
    main()