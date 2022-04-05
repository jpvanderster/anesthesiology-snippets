#%%
# 
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVC


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
        



def perform_grid_search():
    X,y = make_classification(1000,10)

    df = pd.DataFrame(X)

    steps = Pipeline([('scaler', MinMaxScaler()),
         ('clf', LogisticRegression())])
    param_grid = [{
            'clf':[LogisticRegression()],
            'clf__C':np.logspace(-3,3,7)
        },
        {
            'clf':[SVC()],
            # 'clf__n_estimators':np.linspace(100,1000,10)
        },
        {
            'clf':[RandomForestClassifier(n_estimators=10)],
            'clf__n_estimators':[100,200]
        }]

    grid = GridSearchCV(steps,param_grid, refit='roc_auc', scoring=['roc_auc','accuracy'],n_jobs=-1)
    grid.fit(df,y)
    result = pd.DataFrame(grid.cv_results_)
    
    # [print(c) for c in result.columns]
    result.sort_values(by="rank_test_roc_auc", inplace=True)
    result.to_csv("grid_result.csv",index=False,float_format='%3.2f')
    return result


if __name__ == "__main__":
    r = perform_grid_search()
    print('done')