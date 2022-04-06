from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold as kfold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

        
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

def getData(rows=1000,cols=10):
    return make_classification(rows,cols)
    


def custom_fold(X, y, grid, create_plots=False):
    k = kfold(5, shuffle=False)
    fold = 0
    roc_auc = []
    for train_index, test_index in k.split(X, y):        
        _, y_test = y[train_index], y[test_index]
        _, X_test = X[train_index], X[test_index]
        scores = grid.predict_proba(X_test)
        
        fpr, tpr = roc_curve(y_test, scores[:,1], pos_label=1)
        roc_auc.append(auc(fpr, tpr))

        if create_plots:
            f,ax = plt.subplots(1,1)
            ax.plot(fpr,tpr,label=f"ROC (area = {roc_auc}).")
            ax.legend(loc="lower right")
            f.savefig(f"fold_{fold:03d}.png")
        fold+=1

    if create_plots: plt.show()
    return roc_auc

if __name__ == "__main__":
    # Important to make sure your folds are generated the same way as in the grid as in the ROC generation
    X, y = getData(100,10)
    g = perform_grid_search(X, y)
    auc = custom_fold(X, y, g, create_plots=False)
    print(auc)
    print('done')