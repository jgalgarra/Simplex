"""
@author: Iciar Civantos-Gomez
February 2021

This script builds three predictors (Linear regression, Random Forest and XGBoost) splitting data and trainig sets by YEAR.

Invocation: python ALL_FEATURES_predict_by_year 
Builds the predictor with all feautures, using four years as training set and the other as testing set.
Prediction errors (MSE, RMSE, RSE)  by model and year are stored at results/ALLFEATURES_BY_YEAR.xlsx

If the invocation is  python ALL_FEATURES_predict_by_year n 
then precipitation feature is excluded and results are stored at results/ALLFEATURES_BY_YEAR_NOPRECIP.xlsx

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import os
import sys
import rse
import xgboost

verbose = False

if (len(sys.argv)>2):
    print("ERROR. Usage: ALL_FEATURES_predict_by_year.py [include_precipitation]")
    exit()
include_precip = True
if (len(sys.argv) >1):
  if (sys.argv[1]=='n'):
    include_precip = False

"""

Data merging from two sources (ABIOTIC AND BIOTIC). Features year, month, day, 
plotID, x, y, subplot are removed for prediction

"""

print("=================================================")
print("Predictor with environmental and competition data")
print("=================================================")


error_values_lr = []
error_values_rf = []
error_values_xgb = []

initial_year = 2015
final_year = 2020

environment_train = pd.read_csv('../datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')
competitors_train = pd.read_csv('../datasets/abund_merged_dataset_onlycompetitors.csv', sep=',')

individuals_train = environment_train.merge(competitors_train)
 
num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))

if include_precip:
    col_list = ['year','species', 'individuals', 
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'precip', 'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']
else:
    col_list = ['year','species', 'individuals', 
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']

individuals_train = individuals_train[col_list]
individuals_types = individuals_train.dtypes

"Data Wrangling"

"Species transformed to numeric"
le = LabelEncoder()
le.fit(individuals_train[['species']])
individuals_train[['species']] = le.transform(individuals_train[['species']])


if verbose:
    print(individuals_train.dtypes)


"Feature Importance"

input_features = [column for column in list(individuals_train) if column != 'individuals']
X = individuals_train[input_features]
X['random_noise'] = np.random.normal(size=X.shape[0])
y = individuals_train['individuals']

# RF K-Fold train
print("RF K-Fold train")
classifier = RandomForestRegressor(n_jobs=-1)
cv = cross_validate(estimator=classifier, X=X, y=y, cv=5,return_estimator=True)

feature_importance = {}
for k in range(0, len(cv['estimator'])):
    feature_importance['k_{}'.format(k + 1)] = cv['estimator'][k].feature_importances_
feature_importance = pd.DataFrame(feature_importance, index=X.columns)

feature_importance = feature_importance.mean(axis=1).to_frame('importance') \
        .sort_values('importance', ascending=False)
feature_selection = feature_importance.to_dict()

# Get importance concentration score
importance_concentration = (feature_importance.iloc[1] / feature_importance.iloc[0]).values[0] < 0.10
feature_selection['importance_concentration'] = {'assessment': importance_concentration,
                                                 'score': 1 if importance_concentration else 3}

# Get selected features score
selected_features = feature_importance['importance'] > feature_importance.loc['random_noise'].values[0]
selected_features = feature_importance.index[selected_features].tolist()

feature_importance.reset_index(inplace = True)

outputdir = '../results'
if not os.path.exists(outputdir):
  os.makedirs(outputdir)

"Data normalization"

years_datalist= list(range(initial_year,final_year))
for year_test in years_datalist:
    
    print("Testing year ",year_test)
    print(years_datalist)
    years_train = [i for i in years_datalist if i != year_test]
    print("years_train ",years_train)
    
    variables_to_ignore = ['individuals','year']
    selected_features = [element for element in list(individuals_train) if element not in variables_to_ignore]
    
    individuals_model_train = individuals_train.loc[individuals_train['year'].isin(years_train)][selected_features]
    individuals_model_test = individuals_train.loc[individuals_train['year'] == year_test][selected_features]
    
    std_scaler = StandardScaler()
    
    std_scaler_model = std_scaler.fit(individuals_model_train)
    individuals_model_train = std_scaler_model.transform(individuals_model_train)
    
    std_scaler_model = std_scaler.fit(individuals_model_test)
    individuals_model_test = std_scaler_model.transform(individuals_model_test)

    "Division Train Test"
    

    X_train = pd.DataFrame(data = individuals_model_train, columns = selected_features)
    y_train = individuals_train.loc[individuals_train['year'].isin(years_train)]["individuals"]
    
    X_test = pd.DataFrame(data = individuals_model_test, columns = selected_features)
    y_test = individuals_train.loc[individuals_train['year'] == year_test]["individuals"]
    
    
    "Algorithms and evaluation"
    
    "Linear Regression"
    
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    
    predictions_lr = reg.predict(X_test)
    
    rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, predictions_lr))
    mse_lr = mean_squared_error(y_test,predictions_lr)
    rse_lr = rse.calc_rse(y_test,predictions_lr)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_lr,rmse_lr,rse_lr))
    error_values_lr.append((year_test,mse_lr,rmse_lr,rse_lr))
    
    
    "Random Forest"
    
    n_estimators = [100, 150]
    max_features = ['auto']
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features}
    
    seed_value = 4
    random.seed(seed_value)
    
    
    regr = RandomForestRegressor( n_jobs = -1)
    regr_random = RandomizedSearchCV(estimator = regr, 
                                     param_distributions = random_grid, 
                                     cv = 7, verbose=2, n_jobs = -1)
    
    regr_random.fit(X_train,y_train)
    predictions_rf = regr_random.best_estimator_.predict(X_test)
    
    
    rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, predictions_rf))
    mse_rf = mean_squared_error(y_test,predictions_rf)
    rse_rf = rse.calc_rse(y_test,predictions_rf)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf,rmse_rf,rse_rf))
    error_values_rf.append((year_test,mse_rf,rmse_rf,rse_rf))
    
    
    "XGBoost Regressor"
    
    
    xgb = xgboost.XGBRegressor()
    xgb.fit(X_train,y_train)
    
    predictions_xgb = xgb.predict(X_test)
    
    rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, predictions_xgb))
    mse_xgb = mean_squared_error(y_test,predictions_xgb)
    rse_xgb = rse.calc_rse(y_test,predictions_xgb)
    
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_xgb,rmse_xgb,rse_xgb))
    
    error_values_xgb.append((year_test,mse_xgb,rmse_xgb,rse_xgb))
    
    if include_precip:
        prstr = ""
    else:
        prstr = "_NOPRECIP"
    with xlsxwriter.Workbook(outputdir+'/ALLFEATURES_BY_YEAR'+prstr+'.xlsx') as workbook:
        worksheet = workbook.add_worksheet('Linear Regressor')
        worksheet.write_row(0, 0, ['YEAR','MSE','RMSE','RSE'])
        for row_num, data in enumerate(error_values_lr):
            worksheet.write_row(row_num + 1, 0, data)
        
        worksheet = workbook.add_worksheet('Random Forest')
        worksheet.write_row(0, 0, ['YEAR','MSE','RMSE','RSE'])
        for row_num, data in enumerate(error_values_rf):
            worksheet.write_row(row_num + 1, 0, data)
        
        worksheet = workbook.add_worksheet('XGBoost')   
        worksheet.write_row(0, 0, ['YEAR','MSE','RMSE','RSE'])
        for row_num, data in enumerate(error_values_xgb):
            worksheet.write_row(row_num + 1, 0, data)