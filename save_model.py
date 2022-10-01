# Data Manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Sklearn
from sklearn.linear_model import LinearRegression # for building a linear regression model
from sklearn.svm import SVR # for building SVR model
from sklearn.preprocessing import MinMaxScaler

# Visualizations
import plotly.graph_objects as go # for data visualization
import plotly.express as px # for data visualization


X = pd.read_csv('./datasets/3_512_x_main.csv')
y = pd.read_csv('./datasets/3_512_y_main.csv')


model = SVR(C=10) # set kernel and hyper parameters
svr = model.fit(X, y)


# loading dependency

import joblib

# saving our model # model - model , filename-model_jlib
joblib.dump(svr , 'results/model.pkl')