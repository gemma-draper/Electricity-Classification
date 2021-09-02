#%%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


def test_train_split(X, y, test_size=0.2):
    """
    Split X and y data into traing set and test set.
    The default test set proportion is 0.2.
    Inputs: 
        X: 2D feature array.
        y: 1D label array.
    Outputs: 
        X_train, X_test, y_train, y_test
    """
    idx = 0
    length_of_X = len(X)
    y_test = []
    X_test = []
    
    while  idx < length_of_X*test_size:
        random_number_gen = np.random.randint(low=0, high=len(X))
        y_test.append(y[random_number_gen])
        X_test.append(X[random_number_gen])
        X = np.delete(X, random_number_gen, axis=0)
        y = np.delete(y, random_number_gen, axis=0)
        idx += 1
    return X, np.array(X_test), y, np.array(y_test)

def standardise(X_train, X_val, X_test):
    """
    Standardise train, val and test feature sets using statistics from the training set.
    Inputs:  - X_train, X_val, X_test.
    Outputs: - standardised X_train, X_val, X_test.
    """
    mean = np.mean(X_train, axis=0) # get column mean
    print(mean.shape)
    std = np.std(X_train, axis=0) # get column std
    print(std.shape)
    X_list = [X_train, X_val, X_test]
    for X_ in X_list:
        X_ -= mean
        X_ /= std

    return X_train, X_val, X_test


def get_data_split_and_standardise(): 
    """
    Function to import UN electricity data, split into sets and standardise using
    training data statistics.
    Inputs: None
    Outputs: X_train, X_val, X_test, y_train, y_val, y_test
    """   
    np.random.seed(2021)
    data = pd.read_csv(
        '../data/simple_df.csv',
        index_col='Country or Area Code'
    )
    X = data.drop('Label', axis=1).to_numpy() # Drop the label to get features (X)
    y = data['Label'].astype('category').cat.codes.to_numpy() # Convert categories to codes to get features (y)

    # separate the test data
    X_train, X_test, y_train, y_test = test_train_split(X, y)

     # split data into train and val sets.
    X_train, X_val, y_train, y_val = test_train_split(X_train, y_train)

    # standardise the data
    X_train, X_val, X_test = standardise(X_train, X_val, X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__=="__main__":
    # get data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_split_and_standardise()
    # define classifiers
    names = [
        "Nearest Neighbors", 
        "Linear SVM", 
        "Gaussian Process",
        "Decision Tree", 
        "Random Forest", 
        "AdaBoost",
        "Naive Bayes"
    ]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
    ]

    # init an empty dict for scores
    model_scores = {name: {
        'train': [],
        'val': []
        } for name in names}

    # run each model 50 times. We will score the model on the 
    # training and val sets each time, and then average the scores 
    # before plotting.
    for i in range(50):
        # iterate through models
        for name, model in zip(names, classifiers):
            print(f"EVALUATING {name}.")
            # fit the model
            model.fit(X_train, y_train)
            # score the model
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            # append scores to lists
            model_scores[name]['train'].append(train_score)
            model_scores[name]['val'].append(val_score)
    # calculate average model scores over 50 runs
    ave_model_scores = {}
    for name in model_scores:
        ave_model_scores[name] = {
            'train': np.mean(model_scores[name]['train']),
            'validation': np.mean(model_scores[name]['val'])
        }
       
    # plot a bar chart
    # make it a df for easy plotting
    df_scores = pd.DataFrame(ave_model_scores).transpose()
    plt.rcdefaults()
    df_scores.plot(kind='bar')
