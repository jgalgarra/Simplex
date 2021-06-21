"""
@author: Iciar Civantos
February 2021

This script builds as many predictors as the number of experiments set by the contents of `experiments.txt`
First, competitor species are predicted using abiotic features, using Random Forest. Then,
abiotic features and predicted competitors are joined to train the models: linear regression, Random Forest and XGBoost

Prediction errors (MSE, RMSE, RSE) by model and experiment are stored as individual sheets at 
results/TWOSTEP_N.xlsx where N stands for the number of experiments. 

Invocation: python TWOSTEP_predictor.py

CAUTION: This script is CPU-intensive. Running 100 experiments take hours.

"""

import random
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import math
import seaborn as sns
sns.set(color_codes=True)
import xlsxwriter
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
import xgboost
import verde as vd
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
import xgboost

import os
import rse


print("Two-step predictor")
print("======================")

environment_train = pd.read_csv('../datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')
competitors_train = pd.read_csv('../datasets/abund_merged_dataset_onlycompetitors.csv', sep=',')
coord_plot = pd.read_csv('../datasets/coord_plot.csv', sep=';')

conditions = environment_train.merge(competitors_train)

coord_plot["plotID"] = coord_plot["Plot"].apply(str) + coord_plot["Subplot"].apply(lambda x: '_'+x)
conditions = conditions.merge(coord_plot.drop(columns = ["Plot","Subplot"]), on = "plotID", how = "inner")
 

num_rows = len(conditions)
num_cols = len(conditions.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))


col_list = ['species', 'individuals',
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'precip', 'present', 'x', 'y',
       'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']

train_list = ['species', 'individuals', 
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'precip', 'present','x','y']

conditions = conditions[col_list]

conditions_types = conditions.dtypes

"Data Wrangling"

"Transformamos la variable species a numérica"
le = LabelEncoder()
le.fit(conditions[['species']])
conditions[['species']] = le.transform(conditions[['species']])


"Transformamos la variable present a numérica"
le = LabelEncoder()
le.fit(conditions[['present']])
conditions[['present']] = le.transform(conditions[['present']])



perc_0s = round(len(np.where(conditions[['present']] == 0)[0])/num_rows * 100,2)
perc_1s = round(len(np.where(conditions[['present']] == 1)[0])/num_rows * 100,2)

print("===============================================")
print("Original proportion of cases: "+str(perc_0s)+"% of 0s"+\
      " and "+str(perc_1s)+"% of 1s")
print("===============================================")


"Estandarizacion de los datos"

conditions_model_train = conditions[train_list]

std_scaler = StandardScaler()
std_scaler_model = std_scaler.fit(conditions_model_train)
conditions_model_train = std_scaler_model.transform(conditions_model_train)


error_values_lr = []
error_values_rf = []
error_values_xgb = []

outputdir = '../results'
if not os.path.exists(outputdir):
  os.makedirs(outputdir)

with open('experiments.txt', 'r') as g:
  nexper = int(g.readlines()[0])
  
seed_value = 4
random.seed(seed_value)
  
print("nexper",nexper)
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

    y_pred = {}
    rmse_rf = {}
    rse_rf = {}
    
    features_to_pred = ['BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
           'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
           'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']
    
    X = pd.DataFrame(data = conditions_model_train, columns = train_list)
    X[['present']] = conditions[['present']]
    y = conditions[features_to_pred]
    
    X_train_species, X_test_species, y_train_species, y_test_species = train_test_split(X, y, train_size= 0.8)
    "Parámetros Random Forest"
    
    n_estimators = [100,150]
    max_features = ['auto']
    #Grid Search
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features}
    
    
    
    for i in range(0, len(features_to_pred)):
    
        variables_to_ignore = features_to_pred[i]
        print("--------------TARGET "+str(variables_to_ignore))
        
        "Division Train Test"
        
        
        X_train = X_train_species
        y_train = y_train_species[variables_to_ignore]
        
        X_test = X_test_species
        y_test = y_test_species[variables_to_ignore]
        
            
        "Algoritmos y Evaluación"
        
        "Random Forest"
        
        rf = RandomForestRegressor(n_jobs = -1)
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 7, verbose=2, n_jobs = -1)
        
        rf_random.fit(X_train,y_train)
        predictions_rf = rf_random.best_estimator_.predict(X_test)
        
        y_pred[variables_to_ignore] = predictions_rf
        rmse_rf[variables_to_ignore] = np.sqrt(metrics.mean_squared_error(y_test, predictions_rf))
        mse_rf = mean_squared_error(y_test,predictions_rf)
        rse_rf = rse.calc_rse(y_test,predictions_rf)
        #print("RMSE: "+str(rmse_rf[variables_to_ignore]))
        print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf,rmse_rf[variables_to_ignore],rse_rf))
    
    
    
    
    "Utilizamos los resultados para predecir individuals"
    
    features_to_pred = ['individuals']
    selected_features = [element for element in col_list if element not in features_to_pred]
    
    new_X = X_test_species.reset_index().drop(['index'], axis = 1)
    y_predictions = pd.DataFrame.from_dict(y_pred)
    y_predictions = y_predictions.applymap(lambda x: math.floor(x))
    X_individuals = new_X.join(y_predictions)[selected_features]
    
    y_individuals = conditions[features_to_pred].iloc[y_test_species.index].reset_index().drop(['index'], axis = 1)
    
    data = X_individuals.join(y_individuals)
        
    sm = SMOTE(random_state=42)
    data, y_res = sm.fit_resample(data[['species', 'individuals',
        'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
        'k', 'na', 'precip', 'x', 'y', 
        'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
        'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
        'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']], data[['present']])
    data = data.join(y_res)
    
    
    X_ind = data[selected_features]
    y_ind = data['individuals']
    
    "Blocked KFold"
    
    pred_values_lr = np.array([])
    pred_values_rf = np.array([])
    pred_values_xgb = np.array([])
    index_test = []
    
    kfold = vd.BlockKFold(spacing = 0.5, n_splits=4, shuffle=True)
    
    
    "Algorithms and evaluation"
    
    reg = LinearRegression()
    rf = RandomForestRegressor(random_state= seed_value, n_jobs = -1)
    xgb = xgboost.XGBRegressor()
    
    for train, test in kfold.split(np.array(X_ind[['x','y']])):
        
        index_test = np.concatenate((index_test, test), axis=None)
        
        reg.fit(X_ind.iloc[train],y_ind.iloc[train])
        rf.fit(X_ind.iloc[train],y_ind.iloc[train])
        xgb.fit(X_ind.iloc[train],y_ind.iloc[train])
        
        predictions_lr = reg.predict(X_ind.iloc[test])
        pred_values_lr = np.concatenate((pred_values_lr, predictions_lr), axis=None)
        
        predictions_rf = rf.predict(X_ind.iloc[test])
        pred_values_rf = np.concatenate((pred_values_rf, predictions_rf), axis=None)
        
        predictions_xgb = xgb.predict(X_ind.iloc[test])
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
    
    rmse_lr = np.sqrt(metrics.mean_squared_error(y_ind, avg_results_lr))
    mse_lr = mean_squared_error(y_ind,avg_results_lr)
    rse_lr = rse.calc_rse(y_ind,avg_results_lr.prediction)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_lr,rmse_lr,rse_lr))
    error_values_lr.append((mse_lr,rmse_lr,rse_lr))
    
      
    rmse_rf_final = np.sqrt(metrics.mean_squared_error(y_ind, avg_results_rf))
    mse_rf_final = mean_squared_error(y_ind,avg_results_rf)
    rse_rf_final = rse.calc_rse(y_ind,avg_results_rf.prediction)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf_final,rmse_rf_final,rse_rf_final))
    error_values_rf.append((mse_rf_final,rmse_rf_final,rse_rf_final))
       
    
    rmse_xgb = np.sqrt(metrics.mean_squared_error(y_ind, avg_results_xgb))
    mse_xgb = mean_squared_error(y_ind,avg_results_xgb)
    rse_xgb = rse.calc_rse(y_ind,avg_results_xgb.prediction)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_xgb,rmse_xgb,rse_xgb))
    error_values_xgb.append((mse_xgb,rmse_xgb,rse_xgb))

with xlsxwriter.Workbook(outputdir+'/TWOSTEP_'+str(nexper)+'_blocked.xlsx') as workbook:
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
