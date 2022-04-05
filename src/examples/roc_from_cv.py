#%%
from sklearn.model_selection import StratifiedKFold as kfold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class ColumnExtractor:
    def fit(self,X,y=None):
        return self

    def transform(self, X):
        if isinstance(X,np.ndarray):
            print("X is numpy array")
            X_cols = X[:,self.columns]
        elif isinstance(X,pd.DataFrame):
            X_cols = X[self.columns]
            print("X is pandas dataframe")
        else:
            raise("X should be numpy array or pandas dataframe")
        return X_cols

    def __init__(self,columns=None):
        self.columns = columns
        
def perform_grid_search(X,y):
    
    steps = Pipeline([  ('scaler', MinMaxScaler()),
                        ('clf', LogisticRegression())
                    ])
    
    param_grid = [{
                    'clf':[LogisticRegression()],
                    'clf__C':np.logspace(-3,3,7)
                },
                ]

    grid = GridSearchCV(steps, param_grid, refit='roc_auc', scoring=['roc_auc','accuracy'], n_jobs=-1)
    grid.fit(pd.DataFrame(X),y)
    result = pd.DataFrame(grid.cv_results_)
    
    # [print(c) for c in result.columns]
    result.sort_values(by="rank_test_roc_auc", inplace=True)
    result.to_csv("roc_grid_result.csv",index=False,float_format='%3.2f')
    return grid

def getData():
    return make_classification(1000,10)
    
from sklearn.metrics import roc_curve,auc

def custom_fold(X,y,grid):
    k = kfold(5)
    for train_index, test_index in k.split(X, y):
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X[train_index], X[test_index]
        grid.predict(X_test)
        scores = grid.predict_proba(X_test)
        
        fpr, tpr, thresholds = roc_curve(y_test, scores[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        # plt.plot(fpr,tpr,label=f"ROC curve (area = {roc_auc}).")
        # plt.legend(loc="lower right")

        # plt.waitforbuttonpress()
        # plt.close()
    return k

if __name__ == "__main__":
    X, y = getData()

    g = perform_grid_search(X,y)

    k = custom_fold(X,y,g)

    print('done')