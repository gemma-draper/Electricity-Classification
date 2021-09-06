#%%
import numpy as np
import pandas as pd
# import models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV


class LogisticRegressionModel:
    def __init__(self, X_train, X_val, y_train, y_val, penalty='none'):

        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        self.results = []

        self.model = LogisticRegression(penalty=penalty)
        train_score, val_score = self.fit_model()
        result = {
            'name': 'Logistic Regression',
            'penalty': penalty,
            'training score': train_score,
            'validation score': val_score,
            }
        self.results.append(result)
    
    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
        train_score = self.model.score(self.X_train, self.y_train)
        val_score = self.model.score(self.X_val, self.y_val)
        return train_score, val_score


class KNN:
    def __init__(self, hyperparams, X_train, X_val, y_train, y_val):

        self.hyperparams = hyperparams
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        self.results = []

        for param in self.hyperparams['n_neighbors']:
            self.model = KNeighborsClassifier(n_neighbors=param)
            train_score, val_score = self.fit_model()
            result = {
                'name': 'KNN',
                'n_neighbors': param,
                'training score': train_score,
                'validation score': val_score,
                }
            self.results.append(result)
    
    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
        train_score = self.model.score(self.X_train, self.y_train)
        val_score = self.model.score(self.X_val, self.y_val)
        return train_score, val_score

class SVM:
    def __init__(self, params, X_train, X_val, y_train, y_val):

        self.params = params
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        self.results = []

        for kernel in self.params['kernel']:
            for C in self.params['C']:
                self.model = SVC(kernel=kernel, C=C)
                train_score, val_score = self.fit_model()
                result = {
                    'name': 'SVM',
                    'kernel': kernel,
                    'C': C,
                    'training score': train_score,
                    'validation score': val_score,
                    }
                self.results.append(result)
    
    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
        train_score = self.model.score(self.X_train, self.y_train)
        val_score = self.model.score(self.X_val, self.y_val)
        return train_score, val_score
        

class GaussianProcess:
    def __init__(self, hyperparams, X_train, X_val, y_train, y_val):

        self.hyperparams = hyperparams
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        self.results = []

        for kernel in self.hyperparams['kernel']:
            self.model = GaussianProcessClassifier(1.0 * kernel(1.0))
            train_score, val_score = self.fit_model()
            result = {
                'name': 'GaussianProcess',
                'kernel': kernel,
                'training score': train_score,
                'validation score': val_score,
                }
            self.results.append(result)

    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
        train_score = self.model.score(self.X_train, self.y_train)
        val_score = self.model.score(self.X_val, self.y_val)
        return train_score, val_score
        

class DecisionTree:
    def __init__(self) -> None:
        
        DecisionTreeClassifier(max_depth=5)

class RandomForest:
    def __init__(self, hyperparams, X_train, X_val, y_train, y_val):

        self.hyperparams = hyperparams
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        self.results = []

        for max_depth in self.hyperparams['max_depth']:
            for n_estimators in self.hyperparams['n_estimators']:
                for max_features in self.hyperparams['max_features']:
                    self.model = RandomForestClassifier(
                        max_depth=max_depth, 
                        n_estimators=n_estimators, 
                        max_features=max_features
                        )
                    train_score, val_score = self.fit_model()
                    result = {
                        'name': 'RandomForest',
                        'max_depth': max_depth,
                        'n_estimators': n_estimators,
                        'max_features': max_features,
                        'training score': train_score,
                        'validation score': val_score,
                        }
                    self.results.append(result)

    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
        train_score = self.model.score(self.X_train, self.y_train)
        val_score = self.model.score(self.X_val, self.y_val)
        return train_score, val_score

class AdaBoost:
    def __init__(self, hyperparams, X_train, X_val, y_train, y_val):

        self.hyperparams = hyperparams
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        self.results = []

    
        for n_estimators in self.hyperparams['n_estimators']:
            self.model = AdaBoostClassifier(n_estimators=n_estimators)
            train_score, val_score = self.fit_model()
            result = {
                'name': 'AdaBoost',
                'n_estimators': n_estimators,
                'training score': train_score,
                'validation score': val_score,
                }
            self.results.append(result)

    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
        train_score = self.model.score(self.X_train, self.y_train)
        val_score = self.model.score(self.X_val, self.y_val)
        return train_score, val_score
        

def cv_grid_search(model_dict, X_train, X_val, y_train, y_val):
    """
    Grid search using the sklearn GridSearchCV.
    """
    X = np.append(X_train, X_val, axis=0)
    y = np.append(y_train, y_val)

    results = []

    for name in model_dict:
        param_dict = model_dict[name]['params']
        model = model_dict[name]['model']
        classifier = GridSearchCV(
            model, 
            param_dict,
            verbose=2,
            return_train_score=True
            )
        classifier.fit(X, y)   
        results.append(classifier.cv_results_) 
    return pd.DataFrame(results)








if __name__=="__main__":
    names = [
        "Nearest Neighbors", 
        "SVM", 
        "Gaussian Process",
        "Decision Tree", 
        "Random Forest", 
        "AdaBoost"
    ]
    classifiers = [
        KNeighborsClassifier(),
        SVC(),
        GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier()
    ]

    model_dict = {}

    model_dict = {
        name: {
            'model': classifier,
            'params': {}
        } for name, classifier in zip(names, classifiers)}
    # set the params for each model
    model_dict['Nearest Neighbors']['params'] = {
        'n_neighbors': [1 ,2 ,3 ,5 ,10],
        'leaf_size': [10, 20, 30]
    }
    model_dict['SVM']['params'] = {
        'kernel': ['linear', 'rbf'],
        'C': [0.05, 0.1, 0.5, 1]
    }
    model_dict['Gaussian Process']['params'] = {
        'kernel': [1.0 * RBF(1), 2.0 * RBF(2), 5.0 * RBF(5)],
    }
    model_dict['Decision Tree']['params'] = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 7, 10]
    }
    model_dict['Random Forest']['params'] = {
        'max_depth': [5, 7, 10], 
        'n_estimators': [5, 10, 15], 
        'max_features': [1, 2, 3]    
        }
    model_dict['AdaBoost']['params'] = {
        'n_estimators': [20, 35, 50], 
        }

    grid_search(model_dict, X_train, X_val, y_train, y_val)
# %%
