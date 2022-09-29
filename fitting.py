import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet

from utils import fit_result, get_parameters

np.random.seed(42)

dataset = pd.read_csv('./datasets/3_512_x_main.csv')
target = pd.read_csv('./datasets/3_512_y_main.csv')
x_values = dataset.values
y_values = target.values.ravel()

# import fitting_settings.json to retrive optimization hyperparameters for
# raidus = 3, fingerprint bits = 512 dataset
dic = get_parameters(path='./settings/fitting_settings.json', print_dict=False)

 a = fit_result(x_values, y_values, KNeighborsRegressor(algorithm='ball_tree', leaf_size=20, n_neighbors=6, p=1, weights='distance'), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))
b = fit_result(x_values, y_values, Ridge(), random_seeds_start=dic.get("random_seeds_start"),
               random_seeds_stop=dic.get("random_seeds_stop"), random_seeds_step=dic.get("random_seeds_step"))
c = fit_result(x_values, y_values, Lasso(alpha=0.01, max_iter=100000), random_seeds_start=dic.get("random_seeds_start"),
               random_seeds_stop=dic.get("random_seeds_stop"), random_seeds_step=dic.get("random_seeds_step"))
d = fit_result(x_values, y_values, ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=100000),
               random_seeds_start=dic.get("random_seeds_start"), random_seeds_stop=dic.get("random_seeds_stop"),
               random_seeds_step=dic.get("random_seeds_step"))
e = fit_result(x_values, y_values, GradientBoostingRegressor(criterion='mse', loss='huber', min_samples_split=3),
               random_seeds_start=dic.get("random_seeds_start"), random_seeds_stop=dic.get("random_seeds_stop"),
               random_seeds_step=dic.get("random_seeds_step"))
f = fit_result(x_values, y_values, RandomForestRegressor(bootstrap=False, max_features='sqrt', min_samples_split=4),
               random_seeds_start=dic.get("random_seeds_start"), random_seeds_stop=dic.get("random_seeds_stop"),
               random_seeds_step=dic.get("random_seeds_step"))
g = fit_result(x_values, y_values, AdaBoostRegressor(learning_rate=1), random_seeds_start=dic.get("random_seeds_start"),
               random_seeds_stop=dic.get("random_seeds_stop"), random_seeds_step=dic.get("random_seeds_step"))
h = fit_result(x_values, y_values, ExtraTreesRegressor(), random_seeds_start=dic.get("random_seeds_start"),
               random_seeds_stop=dic.get("random_seeds_stop"), random_seeds_step=dic.get("random_seeds_step"))
i = fit_result(x_values, y_values, DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=9),
               random_seeds_start=dic.get("random_seeds_start"), random_seeds_stop=dic.get("random_seeds_stop"),
               random_seeds_step=dic.get("random_seeds_step"))
j = fit_result(x_values, y_values, SVR(C=10), random_seeds_start=dic.get("random_seeds_start"),
               random_seeds_stop=dic.get("random_seeds_stop"), random_seeds_step=dic.get("random_seeds_step"))

df = pd.DataFrame(list(zip(b, c, d, e, f, g, h, i, j)),
                  index=['The average train score of all random states', 'The average test score of all random states',
                         'The average train rmse of all random states', 'The average test rmse of all random states',
                         'The train score std', 'The test score std', 'The train rmse std', "The test rmse std"],
                  columns=["Ridge", 'Lasso', "ElasticNet", "GradientBoostingRegressor", "RandomForestRegressor",
                           "AdaBoostRegressor", "ExtraTreesRegressor", "DecisionTreeRegressor", "SVR"])

df.to_csv('results/Scores.csv')
print("model evalution scores saved as: results/Scores.csv ")
