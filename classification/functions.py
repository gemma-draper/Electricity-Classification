#%%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import models
from sklearn.gaussian_process.kernels import RBF
import itertools

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
    std = np.std(X_train, axis=0) # get column std
    
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

def grid_search(hyperparameters):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

def get_best(results):

    df_results = pd.DataFrame(results)             
    best_result = df_results[df_results['validation score'] == df_results['validation score'].max()].iloc[0].to_json()
    return best_result

def train_all_models(model_dict, X_train, X_val, y_train, y_val):
    """
    Takes dict of all models and hyperparameter variations. Runs a grid search.
    Returns dict of best hyperparameter fit for each model.
    """
    resutls = {}
    best_results = {}
    for name, model_and_params in model_dict.items():
        print(f"EVALUATING {name}.")
        hyperparams = model_and_params['params']

        if name == "Nearest Neighbors":
            current_model = models.KNN(hyperparams, X_train, X_val, y_train, y_val)
        elif name == "SVM":
            current_model = models.SVM(hyperparams, X_train, X_val, y_train, y_val)
        elif name == "Random Forest":
            current_model = models.RandomForest(hyperparams, X_train, X_val, y_train, y_val)
        elif name == "AdaBoost":
            current_model = models.AdaBoost(hyperparams, X_train, X_val, y_train, y_val)
        else: print("Model unrecognised.")
        # # score the model
        resutls[name] = current_model.results
        best_results[name] = get_best(current_model.results)
    return best_results


if __name__=="__main__":
    # get data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_split_and_standardise()
    # define classifiers
    hyperparams = {}
    names = [
        "Nearest Neighbors", 
        "SVM", 
        "Random Forest", 
        "AdaBoost"
    ]
    classifiers = [
        models.KNeighborsClassifier(),
        models.SVC(),
        models.RandomForestClassifier(),
        models.AdaBoostClassifier()
    ]

    model_dict = {}

    model_dict = {
        name: {
            'model': classifier,
            'params': {}
        } for name, classifier in zip(names, classifiers)}
    # set the params for each model
    model_dict['Nearest Neighbors']['params'] = {
        'n_neighbors': [1 ,2 ,3 ,5 ,10]
    }
    model_dict['SVM']['params'] = {
        'kernel': ['linear', 'rbf'],
        'C': [0.05, 0.1, 0.5, 1]
    }
    # model_dict['Gaussian Process']['params'] = {
    #     'kernel': [1.0 * RBF(1), 2.0 * RBF(2), 5.0 * RBF(5)],
    # }
    # model_dict['Decision Tree']['params'] = {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [5, 7, 10]
    # }
    model_dict['Random Forest']['params'] = {
        'max_depth': [5, 7, 10], 
        'n_estimators': [5, 10, 15], 
        'max_features': [1, 2, 3]    
        }
    model_dict['AdaBoost']['params'] = {
        'n_estimators': [20, 35, 50], 
        }

    # models.grid_search(model_dict, X_train, X_val, y_train, y_val)

    # init an empty dict for scores
    model_scores = {name: {
        'train': [],
        'val': []
        } for name in names}

    # run each model 50 times. We will score the model on the 
    # training and val sets each time, and then average the scores 
    # before plotting.
    


            # train_score = model.score(X_train, y_train)
            # val_score = model.score(X_val, y_val)
            # # append scores to lists
            # model_scores[name]['train'].append(train_score)
            # model_scores[name]['val'].append(val_score)

    # # calculate average model scores over 50 runs
    ave_model_scores = {}
    # for name in model_scores:
    #     ave_model_scores[name] = {
    #         'train': np.mean(model_scores[name]['train']),
    #         'validation': np.mean(model_scores[name]['val'])
        # }
       
    # plot a bar chart
    # make it a df for easy plotting
    df_scores = pd.DataFrame(ave_model_scores).transpose()
    plt.rcdefaults()
    df_scores.plot(kind='bar')
#%%
# %%
