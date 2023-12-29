from sklearn.model_selection import GridSearchCV
import numpy as np
from lightgbm import *
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

metadata_path = Path("/projects/foundation/eulp/v1.1.0/BuildingsBench/metadata_dev")

X_train = np.load(metadata_path / "comcap_tune_X_train.npz")["data"]
Y_train = np.load(metadata_path / "comcap_tune_Y_train.npz")["data"]
X_val = np.load(metadata_path / "comcap_tune_X_val.npz")["data"]
Y_val = np.load(metadata_path / "comcap_tune_Y_val.npz")["data"]

parameters = {'metric': ["rmse"], 
              'learning_rate': [0.01, 0.02, 0.05, 0.1],
              'max_depth': [10, 20, 50, 100],
              'num_leaves': [20, 50, 100],
              'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
              'reg_lambda': [0.01, 0.1],
              'reg_alpha':  [0.01, 0.1]
             }

grid_cv = GridSearchCV(estimator=LGBMRegressor(categorical_feature=[i for i in range(12, 32)]), param_grid=parameters, n_jobs=10)
grid_cv.fit(X_train, Y_train[:, 0])

import pickle

with open('grid_search_cv.pkl', 'wb') as f:
    pickle.dump(grid_cv, f, pickle.HIGHEST_PROTOCOL)

def normalize_rmse(Y_pred, Y_true):
    rmse = np.sqrt(np.mean((Y_pred - Y_true) ** 2)) / np.mean(Y_true)
    return rmse

lgbm = grid_cv.best_estimator_
rmse_train = normalize_rmse(lgbm.predict(X_train), Y_train[:, 0])
rmse_val   = normalize_rmse(lgbm.predict(X_val), Y_val[:, 0])

print("train rmse", rmse_train)
print("val rmse", rmse_val)