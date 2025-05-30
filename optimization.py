import numpy as np
import pandas as pd
from utils import pipeline_optim

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(filename="results/Best_parameters.log", level=logging.INFO, format= '%(asctime)s %(name)s %(levelname)s %(message)s')


np.random.seed(42)
dataset = pd.read_csv('./datasets/3_512_x_main.csv')
target = pd.read_csv('./datasets/3_512_y_main.csv')

# generate optimization hyperparameters of algorithms

knn = pipeline_optim(dataset, target, random_seeds = 42, mode = 'knn')
ridge = pipeline_optim(dataset, target, random_seeds = 42, mode = 'ridge')
lasso = pipeline_optim(dataset, target, random_seeds = 42, mode = 'lasso')
elastic = pipeline_optim(dataset, target, random_seeds = 42, mode = 'elastic')
gb = pipeline_optim(dataset, target, random_seeds = 42, mode = 'gradientboosting')
rf = pipeline_optim(dataset, target, random_seeds = 42, mode = 'rf')
ada = pipeline_optim(dataset, target, random_seeds = 42, mode = 'adaboost')
extra = pipeline_optim(dataset, target, random_seeds = 42, mode = 'extratrees')
dt = pipeline_optim(dataset, target, random_seeds = 42, mode = 'dt')
svr = pipeline_optim(dataset, target, random_seeds = 42, mode = 'svr')

logging.info(knn)
logging.info(ridge)
logging.info(lasso)
logging.info(elastic)
logging.info(gb)
logging.info(rf)
logging.info(ada)
logging.info(extra)
logging.info(dt)
logging.info(svr)

logging.shutdown()