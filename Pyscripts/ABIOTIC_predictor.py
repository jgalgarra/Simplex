"""
@author: Iciar Civantos
February 2021

This script builds as many predictors as the number of experiments set by the contents of `experiments.txt`
Only abiotic features are included to train the models: linear regression, Random Forest and XGBoost
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
import xgboost
from sklearn.metrics import mean_squared_error
import sys
import rse
import os
verbose = True

if (len(sys.argv)>2):
    print("ERROR. Usage: ABIOTIC_predictor.py [include_precipitation]")
    exit()
include_precip = True
if (len(sys.argv) >1):
  if (sys.argv[1]=='n'):
    include_precip = False


print("Predictor with environmental data only")
print("=======================================")

individuals_train = pd.read_csv('../datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')

num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))

if include_precip:
   base_list = ['species','individuals','ph','salinity','precip',
                'cl','co3','c','mo','n','cn','p','ca','mg','k','na']

else:
   base_list = ['species','individuals','ph','salinity',
                'cl','co3','c','mo','n','cn','p','ca','mg','k','na']
col_list = base_list.copy()
individuals_train = individuals_train[col_list]
individuals_types = individuals_train.dtypes

"Data Wrangling"

le = LabelEncoder()
le.fit(individuals_train[['species']])
individuals_train[['species']] = le.transform(individuals_train[['species']])

if verbose:
    print(individuals_train.dtypes)

"Random Forest parameters"

# Number of trees in random forest
n_estimators = [100, 150]
# Number of features to consider at every split
max_features = ['auto']
#Grid Search
random_grid = {'n_estimators': n_estimators,
           'max_features': max_features}

error_values_lr = []
error_values_rf = []
error_values_xgb = []

outputdir = '../results'
if not os.path.exists(outputdir):
  os.makedirs(outputdir)

with open('experiments.txt', 'r') as g:
  nexper = int(g.readlines()[0])
  
print("nexper",nexper)
seed_value = 4
random.seed(seed_value)
for i in range(0, nexper):
    
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
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8)
    print(X_train.columns)
    
    "Algoritmos y Evaluación"
    
    "Linear Regression"
    
    print("Linear Regression")
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    
    predictions_lr = reg.predict(X_test)
    
    rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, predictions_lr))
    mse_lr = mean_squared_error(y_test,predictions_lr)
    rse_lr = rse.calc_rse(y_test,predictions_lr)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_lr,rmse_lr,rse_lr))
    
    error_values_lr.append((mse_lr,rmse_lr,rse_lr))
    
    "Random Forest"
    
    print("Random Forest")
    
    regr = RandomForestRegressor(random_state= seed_value, n_jobs = -1)
    regr_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, cv = 7, verbose=2, n_jobs = -1)
    
    regr_random.fit(X_train,y_train)
    predictions_rf = regr_random.best_estimator_.predict(X_test)
    
    rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, predictions_rf))    
    mse_rf = mean_squared_error(y_test,predictions_rf)
    rse_rf = rse.calc_rse(y_test,predictions_rf)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf,rmse_rf,rse_rf))
    
    error_values_rf.append((mse_rf,rmse_rf,rse_rf))
    
    "XGBoost Regressor"

    xgb = xgboost.XGBRegressor()
    xgb.fit(X_train,y_train)
    
    predictions_xgb = xgb.predict(X_test)
    
    rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, predictions_xgb))
    mse_xgb = mean_squared_error(y_test,predictions_xgb)
    rse_xgb = rse.calc_rse(y_test,predictions_xgb)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_xgb,rmse_xgb,rse_xgb))
    
    error_values_xgb.append((mse_xgb,rmse_xgb,rse_xgb))
    
if include_precip:
    prstr = ""
else:
    prstr = "NOPRECIP_"
with xlsxwriter.Workbook(outputdir+'/ABIOTIC_'+prstr+str(nexper)+'.xlsx') as workbook:
    worksheet = workbook.add_worksheet('Linear Regressor')
    worksheet.write_row(0, 0, ['MSE','RMSE','RSE'])
    for row_num, data in enumerate(error_values_lr):
        worksheet.write_row(row_num + 1, 0, data)
    
    worksheet = workbook.add_worksheet('Random Forest')
    worksheet.write_row(0, 0, ['MSE','RMSE','RSE'])
    for row_num, data in enumerate(error_values_rf):
        worksheet.write_row(row_num + 1, 0, data)
    
    worksheet = workbook.add_worksheet('XGBoost')   
    worksheet.write_row(0, 0, ['MSE','RMSE','RSE'])
    for row_num, data in enumerate(error_values_xgb):
        worksheet.write_row(row_num + 1, 0, data)
