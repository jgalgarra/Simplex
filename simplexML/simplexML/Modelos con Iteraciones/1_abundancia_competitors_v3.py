
"""
Created on Sun Aug  9 11:37:50 2020
@author: Iciar Civantos
This script builds the individuals predictor using species competitors abundance
"""

import random

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set(color_codes=True)

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
import xgboost
import config as cf
import rse
import sys
import warnings
warnings.filterwarnings("ignore")

# verbose = cf.verbose
verbose = True

if (len(sys.argv)>2):
    print("ERROR. Usage: 1_abundancia_competitors.py [present_percentage]")
    exit()
if (len(sys.argv) ==1):
    smote_yn = 'n'
else:
    perc_0s = float(sys.argv[1])
    smote_0s = round(perc_0s/100,2)
    smote_yn = 'y'

print("Predictor with competitor species only")
print("=======================================")
competitors_train = pd.read_csv('datasets/abund_merged_dataset_onlycompetitors.csv', sep=',')
environment_train = pd.read_csv('datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')
individuals_train = environment_train.merge(competitors_train)

num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))


col_list = ['species','present','individuals','BEMA','CETE','CHFU','CHMI','COSQ','FRPU','HOMA','LEMA','LYTR','MEEL','MEPO','MESU','PAIN','PLCO','POMA','POMO','PUPA','RAPE','SASO','SCLA','SOAS','SPRU','SUSP']

individuals_train = individuals_train[col_list]
individuals_types = individuals_train.dtypes

"Data Wrangling"

"Transformamos la variable species a numérica"
le = LabelEncoder()
le.fit(individuals_train[['species']])
individuals_train[['species']] = le.transform(individuals_train[['species']])

le = LabelEncoder()
le.fit(individuals_train[['present']])
individuals_train[['present']] = le.transform(individuals_train[['present']])

perc_0s = round(len(np.where(individuals_train[['present']] == 0)[0])/num_rows * 100,2)
perc_1s = round(len(np.where(individuals_train[['present']] == 1)[0])/num_rows * 100,2)

print("===============================================")
print("Original proportion of cases: "+str(perc_0s)+"% of 0s"+\
      " and "+str(perc_1s)+"% of 1s")
print("===============================================")

# print("===============================================")
# smote_yn = str(input("¿Desea incrementar el %?(y/n): "))

if smote_yn == 'y':
    # print("Inserte nuevo porcentaje")
    # perc_0s = float(input("Introduzca porcentaje de 1s: "))
    smote_0s = round(perc_0s/100,2)
    
    sm = SMOTE(random_state=42,sampling_strategy = smote_0s)
    individuals_train, y_res = sm.fit_resample(individuals_train[['species','individuals','BEMA','CETE','CHFU','CHMI','COSQ','FRPU','HOMA','LEMA','LYTR','MEEL','MEPO','MESU','PAIN','PLCO','POMA','POMO','PUPA','RAPE','SASO','SCLA','SOAS','SPRU','SUSP']], individuals_train[['present']])
    individuals_train = individuals_train.join(y_res)
    
else:
    print("===============================================")
    print("No SMOTE balancing")
    print("===============================================")

if verbose:
    print(individuals_train.dtypes)

"Ver si hay registros duplicados"

if verbose:
    print("Ver si hay registros duplicados")
num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)

individuals_train = individuals_train.drop_duplicates()
num_rows_clean = len(individuals_train)
if verbose:
    print("In this dataset there were {} repeated records".format(num_rows - num_rows_clean))



"Parámetros Random Forest"

# Number of trees in random forest
n_estimators = [100, 150]
# Number of features to consider at every split
max_features = ['auto']
#Grid Search
random_grid = {'n_estimators': n_estimators,
           'max_features': max_features}

error_values = []

for i in range(0, 20):

    "Estandarizacion de los datos"
    
    variables_to_ignore = ['individuals']
    selected_features = [element for element in list(individuals_train) if element not in variables_to_ignore]
    
    individuals_model_train = individuals_train[selected_features]
    
    std_scaler = StandardScaler()
    
    std_scaler_model = std_scaler.fit(individuals_model_train)
    individuals_model_train = std_scaler_model.transform(individuals_model_train)
    
    
    "Division Train Test"
    
    X = pd.DataFrame(data = individuals_model_train, columns = selected_features)
    y = individuals_train.individuals
    
    #X = X.drop(['random_noise'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8)
    print(X_train.columns)
    
    "Algoritmos y Evaluación"
    
    
    
    "Random Forest"
    
    print("Random Forest")
    seed_value = 4
    random.seed(seed_value)
    
    
    regr = RandomForestRegressor( random_state= seed_value,n_jobs = -1)
    # regr = RandomForestRegressor(random_state= seed_value, n_jobs = -1, n_estimators = 150)
    regr_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, cv = 7, verbose=2, n_jobs = -1)
    
    regr_random.fit(X_train,y_train)
    print(regr_random.best_params_)
    predictions_rf = regr_random.best_estimator_.predict(X_test)
    
    
    
    
    rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, predictions_rf))
    
    mse_rf = mean_squared_error(y_test,predictions_rf)
    rse_rf = rse.calc_rse(y_test,mse_rf)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf,rmse_rf,rse_rf))
    
    error_values.append((mse_rf,rmse_rf,rse_rf))