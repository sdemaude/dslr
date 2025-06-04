#It takes as a parameter dataset_train.csv. For the mandatory part, you must use the technique of gradient descent to minimize the error.
#The program generates a file containing the weights that will be used for the prediction.

#genere a file with the weights of the logistic regression model

import calculation_utils as calc
import pandas as pd
import numpy as np
import sys

def normalize(X):
    #means = X.mean()
    #stds = X.std()
    #return (X - means) / stds
    means = calc.ft_mean(X)
    stds = calc.ft_std(X)
    return (X - means) / stds

def sigmoid(z):
    return 1 / (1 + np.exp(-z)) # re-code the exponential function from scratch ?

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -1/m * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) # re-code the log function from scratch ? + the matrix multiplication
    return cost

#def gradient(X, y, theta):

#def logistic_regression(X, y, alpha=0.01, num_iterations=1000):
    # sigmoid function
    # cost function
    # gradient descent to minimize the cost function
    # return theta

def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_file>")
        sys.exit(1)
    dataset_file = sys.argv[1]
    df = pd.read_csv(dataset_file)

    features = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts']
    df = df.dropna(subset=features + ['Hogwarts House'])
    X = df[features].values
    y = df['Hogwarts House'].values

    X = normalize(X)
    X = np.c_[np.ones(X.shape[0]), X]  # ajoute 1 au début de chaque ligne (terme de biais)

    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    #logistic regression

    '''
    1. Charger les données d'entraînement (dataset_train.csv)
    - X = tableau des notes des élèves (features)
    - y = leur vraie maison

    2. Normaliser X (mettre les notes sur une échelle comparable)

    3. Ajouter une colonne de "1" à X (c'est le biais)

    4. Pour chaque maison (Ravenclaw, Gryffindor, etc.) :
    a. Créer un tableau y_binary :
        - 1 si l'élève est dans cette maison
        - 0 sinon
    b. Initialiser les poids θ à 0
    c. Répéter (ex: 1000 fois) :
        i.   Faire la prédiction : h = sigmoid(X * θ)
        ii.  Calculer l'erreur entre h et y_binary
        iii. Calculer le gradient
        iv.  Mettre à jour les poids : θ = θ - alpha * gradient
    d. Sauvegarder θ pour cette maison

    5. Sauvegarder tous les θ dans un fichier weights.csv
        Exemple de format :
            Houses,Weights
            Gryffindor,0.1
            Ravenclaw,0.4
            ...

    '''


if __name__ == '__main__':
    main()