### to save predictions for complete data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import os
from sklearn.svm import SVR
import joblib


import warnings
warnings.filterwarnings("ignore")


import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit_utils import smiles_dataset
from utils import save_dataset


model_load = joblib.load('./models/model.pkl')
database = pd.read_csv('./screening_base/in-vitro_zinc/in-vitro.csv')
screen_database = pd.read_csv('./datasets/screen_results/in-vitro_zinc/in-vitro_bits.csv')

screen_result = model_load.predict(screen_database)
screen_result_fp = pd.DataFrame({'Predictive Results': screen_result})
predictions_on_zinc15 = pd.concat([database, screen_result_fp], axis = 1)


save_dataset(predictions_on_zinc15, path = 'results/', file_name = 'database_result', idx = False)
