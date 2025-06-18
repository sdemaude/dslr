import numpy as np
import matplotlib.pyplot as plt

def train_visualization(Histories, houses, y_true, y_pred, accuracy):
    fig = plt.figure(figsize=(14, 6))
    
    # Loss curves (left)
    plt.subplot(1, 2, 1)
    for house in houses:
        plt.plot(Histories[house]['Losses'], label=house)
    plt.title('Loss Curves - All Houses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Confusion matrix (right)
    plt.subplot(1, 2, 2)
    n_classes = len(houses)
    conf_matrix = np.zeros((n_classes, n_classes))
    
    for i, true_house in enumerate(houses):
        for j, pred_house in enumerate(houses):
            conf_matrix[i, j] = np.sum((y_true == true_house) & (y_pred == pred_house))
    
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - Accuracy: {accuracy * 100:.2f}%')
    
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, houses, rotation=45)
    plt.yticks(tick_marks, houses)
    
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, f'{int(conf_matrix[i, j])}', 
                    ha="center", va="center", color="black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()
