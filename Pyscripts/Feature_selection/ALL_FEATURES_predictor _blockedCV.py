"""
@author: Iciar Civantos
February 2021

This script builds as many predictors as the number of experiments set by the contents of `experiments.txt`
All features are included to train the models: linear regression, Random Forest and XGBoost
For each experiment, there is a random training/testing split.

Prediction errors (MSE, RMSE, RSE) by model and experiment are stored as individual sheets at 
results/ALLFEATURES_N.xlsx where N stands for the number of experiments. 

Invocation: python ALLFEATURES_predictor.py
If the invocation is python ALLFEATURES_predictor.py n, then precipitation feature is excluded and the results
file is called ALLFEATURES_NOPRECIP_N.xlsx

"""
import random
import xlsxwriter
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import seaborn as sns
sns.set(color_codes=True)

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
import xgboost
import verde as vd
from sklearn.metrics import mean_squared_error
import os
import xgboost
import rse
verbose = False
import sys

if (len(sys.argv)>2):
    print("ERROR. Usage: ALL_FEATURES_predictor.py [include_precipitation]")
    exit()
include_precip = True
if (len(sys.argv) >1):
  if (sys.argv[1]=='n'):
    include_precip = False

print("Predictor with environmental and competition data")
print("=================================================")


environment_train = pd.read_csv('../../datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')
competitors_train = pd.read_csv('../../datasets/abund_merged_dataset_onlycompetitors.csv', sep=',')
coord_plot = pd.read_csv('../../datasets/coord_plot.csv', sep=';')

individuals_train = environment_train.merge(competitors_train)

coord_plot["plotID"] = coord_plot["Plot"].apply(str) + coord_plot["Subplot"].apply(lambda x: '_'+x)
individuals_train = individuals_train.merge(coord_plot.drop(columns = ["Plot","Subplot"]), on = "plotID", how = "inner")
 

num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))

if include_precip:
   col_list = ['species', 'individuals',
       'ph', 'salinity', 'cl', 'co3', 'c', 'p', 'ca', 'mg',
       'k', 'na', 'precip', 'x', 'y', 'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 
       'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']
else:
   col_list = ['species', 'individuals',
       'ph', 'salinity', 'cl', 'co3', 'c', 'p', 'ca', 'mg',
       'k', 'na', 'x', 'y',  'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 
       'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']
individuals_train = individuals_train[col_list]
individuals_types = individuals_train.dtypes

"Data Wrangling"

"Transformamos la variable species a numérica"
le = LabelEncoder()
le.fit(individuals_train[['species']])
individuals_train[['species']] = le.transform(individuals_train[['species']])


if verbose:
    print(individuals_train.dtypes)

outputdir = '../../results'
if not os.path.exists(outputdir):
  os.makedirs(outputdir)

with open('experiments.txt', 'r') as g:
  nexper = int(g.readlines()[0])
  
print("nexper",nexper)
seed_value = 4
random.seed(seed_value)

"Estandarizacion de los datos"

variables_to_ignore = ['individuals']
selected_features = [element for element in list(individuals_train) if element not in variables_to_ignore]

individuals_model_train = individuals_train[selected_features]

std_scaler = StandardScaler()

std_scaler_model = std_scaler.fit(individuals_model_train)
individuals_model_train = std_scaler_model.transform(individuals_model_train)


"Division X and y"

X = pd.DataFrame(data = individuals_model_train, columns = selected_features)
y = individuals_train.individuals

error_values_lr = []
error_values_rf = []
error_values_xgb = []

for i in range(0, 100):
    
    print("============================================================================")
    print("============================================================================")
    print("============================================================================")
    print("============================================================================")
    print("============================================================================")
    print("======================== ITER: "+str(i)+" ==================================")
    print("============================================================================")
    print("============================================================================")
    print("============================================================================")
    print("============================================================================")

    
    "Blocked KFold"
    
    pred_values_lr = np.array([])
    pred_values_rf = np.array([])
    pred_values_xgb = np.array([])
    index_test = []
    
    kfold = vd.BlockKFold(spacing = 0.5, n_splits=4, shuffle=True)
    
    
    "Predictions"
    
    reg = LinearRegression()
    rf = RandomForestRegressor(random_state= seed_value, n_jobs = -1)
    xgb = xgboost.XGBRegressor()
    
    for train, test in kfold.split(np.array(X[['x','y']])):
        index_test = np.concatenate((index_test, test), axis=None)
        
        reg.fit(X.iloc[train],y.iloc[train])
        rf.fit(X.iloc[train],y.iloc[train])
        xgb.fit(X.iloc[train],y.iloc[train])
        
        predictions_lr = reg.predict(X.iloc[test])
        pred_values_lr = np.concatenate((pred_values_lr, predictions_lr), axis=None)
        
        predictions_rf = rf.predict(X.iloc[test])
        pred_values_rf = np.concatenate((pred_values_rf, predictions_rf), axis=None)
        
        predictions_xgb = xgb.predict(X.iloc[test])
        pred_values_xgb = np.concatenate((pred_values_xgb, predictions_xgb), axis=None)
    
    "Median prediction"
    
    results_lr = pd.concat([pd.Series(index_test), pd.Series(pred_values_lr)], axis = 1)
    results_lr.columns = ['test_index', 'prediction']
    avg_results_lr = results_lr.groupby('test_index').median()
    
    results_rf = pd.concat([pd.Series(index_test), pd.Series(pred_values_rf)], axis = 1)
    results_rf.columns = ['test_index', 'prediction']
    avg_results_rf = results_rf.groupby('test_index').median()
    
    results_xgb = pd.concat([pd.Series(index_test), pd.Series(pred_values_xgb)], axis = 1)
    results_xgb.columns = ['test_index', 'prediction']
    avg_results_xgb = results_xgb.groupby('test_index').median()
    
    "Error Calculation"
    
    rmse_lr = np.sqrt(metrics.mean_squared_error(y, avg_results_lr))
    mse_lr = mean_squared_error(y,avg_results_lr)
    rse_lr = rse.calc_rse(y,avg_results_lr.prediction)
    
    error_values_lr.append((mse_lr,rmse_lr,rse_lr))     
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_lr,rmse_lr,rse_lr))
    
    rmse_rf = np.sqrt(metrics.mean_squared_error(y, avg_results_rf))    
    mse_rf = mean_squared_error(y,avg_results_rf)
    rse_rf = rse.calc_rse(y,avg_results_rf.prediction)
    
    error_values_rf.append((mse_rf,rmse_rf,rse_rf))
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf,rmse_rf,rse_rf))
    
    rmse_xgb = np.sqrt(metrics.mean_squared_error(y, avg_results_xgb))    
    mse_xgb = mean_squared_error(y,avg_results_xgb)
    rse_xgb = rse.calc_rse(y,avg_results_xgb.prediction)
    
    error_values_xgb.append((mse_xgb,rmse_xgb,rse_xgb))
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_xgb,rmse_xgb,rse_xgb))


with xlsxwriter.Workbook('../../results/ALL_FEATURES_fs_blocked.xlsx') as workbook:
    worksheet = workbook.add_worksheet()
    worksheet.write_row(0, 0, ['MSE','RMSE','RSE'])
    for row_num, data in enumerate(error_values_lr):
        worksheet.write_row(row_num + 1, 0, data)
    
    worksheet = workbook.add_worksheet()
    worksheet.write_row(0, 0, ['MSE','RMSE','RSE'])
    for row_num, data in enumerate(error_values_rf):
        worksheet.write_row(row_num + 1, 0, data)
    
    worksheet = workbook.add_worksheet()   
    worksheet.write_row(0, 0, ['MSE','RMSE','RSE'])
    for row_num, data in enumerate(error_values_xgb):
        worksheet.write_row(row_num + 1, 0, data)