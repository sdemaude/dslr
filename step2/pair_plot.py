import matplotlib.pyplot as plt
import pandas as pd

house_colors = {
    'Gryffindor': 'DarkRed',
    'Ravenclaw': 'RoyalBlue',
    'Hufflepuff': 'Gold',
    'Slytherin': 'DarkGreen'
}

def wrap_label(label, max_len=16):
    return '\n'.join([label[i:i+max_len] for i in range(0, len(label), max_len)])

def main():
    dataset = pd.read_csv('datasets/dataset_train.csv')
    courses = dataset.select_dtypes(include='float').columns
    houses = dataset['Hogwarts House'].unique()

    num_courses = len(courses)
    fig, axes = plt.subplots(num_courses, num_courses, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i in range(num_courses):
        for j in range(num_courses):
            ax = axes[i, j]
            x_col = courses[j]
            y_col = courses[i]

            if i == j:
                for house in houses:
                    subset = dataset[dataset['Hogwarts House'] == house]
                    ax.hist(subset[x_col].dropna(), alpha=0.6, label=house, color=house_colors.get(house, 'gray'))         
            else:
                for house in houses:
                    subset = dataset[dataset['Hogwarts House'] == house]
                    ax.scatter(subset[x_col], subset[y_col], alpha=0.3, s=10, label=house, color=house_colors.get(house, 'gray'))
            
            ax.tick_params(axis='both', labelbottom=False, labelleft=False) 
            if i == num_courses - 1:
                ax.set_xlabel(wrap_label(x_col), fontsize=10, rotation=30)
                ax.xaxis.set_label_coords(0.5, -0.3)
            if j == 0:
                ax.set_ylabel(wrap_label(y_col), fontsize=10, rotation=0)
                ax.yaxis.set_label_coords(-1, 0.3)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10, title='Hogwarts Houses')

    plt.suptitle("Pair Plot of Courses by Hogwarts House", fontsize=16)
    plt.show()

if __name__ == '__main__':
    main()