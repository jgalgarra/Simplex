

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import math


import xlsxwriter
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
import xgboost

import rse


print("Two-step predictor")
print("======================")

environment_train = pd.read_csv('datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')
competitors_train = pd.read_csv('datasets/abund_merged_dataset_onlycompetitors.csv', sep=',')

conditions = environment_train.merge(competitors_train)
 

num_rows = len(conditions)
num_cols = len(conditions.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))


col_list = ['species', 'individuals',
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'precip', 'present',
       'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']

train_list = ['species', 'individuals', 
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'precip', 'present']

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



"Estandarizacion de los datos"

conditions_model_train = conditions[train_list]

std_scaler = StandardScaler()
std_scaler_model = std_scaler.fit(conditions_model_train)
conditions_model_train = std_scaler_model.transform(conditions_model_train)


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
        rse_rf = rse.calc_rse(predictions_rf,mse_rf)
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
        'k', 'na', 'precip',
        'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
        'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
        'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']], data[['present']])
    data = data.join(y_res)
    
    
    X_ind = data[selected_features]
    y_ind = data['individuals']
    
    X_train_individuals, X_test_individuals, y_train_individuals, y_test_individuals = train_test_split(X_ind, y_ind, train_size= 0.8)
    
    "Algoritmos y Evaluación"

    "Linear Regression"
    
    reg = LinearRegression()
    reg.fit(X_train_individuals,y_train_individuals)
    
    predictions_lr = reg.predict(X_test_individuals)
    
    rmse_lr = np.sqrt(metrics.mean_squared_error(y_test_individuals, predictions_lr))
    mse_lr = mean_squared_error(y_test_individuals,predictions_lr)
    rse_lr = rse.calc_rse(y_test_individuals,predictions_lr)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_lr,rmse_lr,rse_lr))
    error_values_lr.append((mse_lr,rmse_lr,rse_lr))
    
    "Random Forest"
    # print("Random Forest")
    seed_value = 4
    # random.seed(seed_value)
    
    rf = RandomForestRegressor(random_state= seed_value, n_jobs = -1, n_estimators = 150)
    rf.fit(X_train_individuals,y_train_individuals)
    predictions_rf = rf.predict(X_test_individuals)
    
    # rf = RandomForestRegressor(n_jobs = -1)
    # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 7, verbose=2, n_jobs = -1)
    
    # rf_random.fit(X_train_individuals,y_train_individuals)
    # predictions_rf = rf_random.best_estimator_.predict(X_test_individuals)
    
    rmse_rf_final = np.sqrt(metrics.mean_squared_error(y_test_individuals, predictions_rf))
    mse_rf_final = mean_squared_error(y_test_individuals,predictions_rf)
    rse_rf_final = rse.calc_rse(y_test_individuals,predictions_rf)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf_final,rmse_rf_final,rse_rf_final))
    error_values_rf.append((mse_rf_final,rmse_rf_final,rse_rf_final))
    
    
    "XGBoost Regressor"

    xgb = xgboost.XGBRegressor()
    xgb.fit(X_train_individuals,y_train_individuals)
    
    predictions_xgb = xgb.predict(X_test_individuals)
    
    rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test_individuals, predictions_xgb))
    mse_xgb = mean_squared_error(y_test_individuals,predictions_xgb)
    rse_xgb = rse.calc_rse(y_test_individuals,predictions_xgb)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_xgb,rmse_xgb,rse_xgb))
    error_values_xgb.append((mse_xgb,rmse_xgb,rse_xgb))

with xlsxwriter.Workbook('competitors.xlsx') as workbook:
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