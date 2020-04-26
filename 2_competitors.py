import pandas as pd
import random
pd.set_option('display.max_colwidth', -1)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



environment_train = pd.read_csv('datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')
competitors_train = pd.read_csv('datasets/abund_merged_dataset_onlycompetitors.csv', sep=',')

conditions = environment_train.merge(competitors_train)
 

num_rows = len(conditions)
num_cols = len(conditions.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))


col_list = ['year', 'species', 'individuals', 'plotID', 'x', 'y',
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'plot', 'subplot', 'precip', 'sum_salinity', 'present',
       'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']

train_list = ['year', 'species', 'individuals', 'plotID', 'x', 'y',
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'plot', 'subplot', 'precip', 'sum_salinity', 'present']

conditions = conditions[col_list]

conditions_types = conditions.dtypes

"Data Wrangling"

"Transformamos la variable species a numérica"
le = LabelEncoder()
le.fit(conditions[['species']])
conditions[['species']] = le.transform(conditions[['species']])

"Transformamos la variable plotID a numérica"
le = LabelEncoder()
le.fit(conditions[['plotID']])
conditions[['plotID']] = le.transform(conditions[['plotID']])

"Transformamos la variable subplot a numérica"
le = LabelEncoder()
le.fit(conditions[['subplot']])
conditions[['subplot']] = le.transform(conditions[['subplot']])

"Transformamos la variable present a numérica"
le = LabelEncoder()
le.fit(conditions[['present']])
conditions[['present']] = le.transform(conditions[['present']])

"Estandarizacion de los datos"

conditions_model_train = conditions[train_list]

std_scaler = StandardScaler()
std_scaler_model = std_scaler.fit(conditions_model_train)
conditions_model_train = std_scaler_model.transform(conditions_model_train)

y_pred = {}
rmse_rf = {}
features_to_pred = ['BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']

for i in range(0, len(features_to_pred)):

    variables_to_ignore = features_to_pred[i]
    print("--------------TARGET "+str(variables_to_ignore))
    
    "Division Train Test"
    
    X = pd.DataFrame(data = conditions_model_train, columns = train_list)
    y = conditions[variables_to_ignore]
    
    X_train = X.iloc[0:29808]
    y_train = y.iloc[0:29808]
    
    X_test = X.iloc[29808::]
    y_test = y.iloc[29808::]
    
        
    "Algoritmos y Evaluación"
    
    "Random Forest"
    rf = RandomForestRegressor(n_jobs = -1)
    
    rf.fit(X_train,y_train)
    predictions_rf = rf.predict(X_test)
    
    y_pred[variables_to_ignore] = predictions_rf
    rmse_rf[variables_to_ignore] = np.sqrt(metrics.mean_squared_error(y_test, predictions_rf))
    print("RMSE: "+str(rmse_rf[variables_to_ignore]))
    

"Utilizamos los resultados para predecir individuals"

features_to_pred = ['individuals']
selected_features = [element for element in col_list if element not in features_to_pred]

new_X = conditions[train_list].iloc[29808::].reset_index()
y_predictions = pd.DataFrame.from_dict(y_pred)

X_individuals = new_X.join(y_predictions).drop(['index'], axis = 1)
y_individuals = conditions[features_to_pred].iloc[29808::]


X_train_individuals, X_test_individuals, y_train_individuals, y_test_individuals = train_test_split(X_individuals, y_individuals, train_size= 0.8)

"Random Forest"

seed_value = 4
random.seed(seed_value)
rf = RandomForestRegressor(random_state= seed_value)

param_grid = {'max_features':['auto', 'log2'], 'n_estimators':[100,150]}
cross_val_rf = GridSearchCV(rf, param_grid, cv = 5)
cross_val_rf.fit(X_train_individuals,y_train_individuals)
predictions_rf = cross_val_rf.predict(X_test_individuals)

rmse_rf_final = np.sqrt(metrics.mean_squared_error(y_test_individuals, predictions_rf))

best_result_rf = cross_val_rf.best_params_
(print("The best random forest has a max_features value of {0} and n_estimators of {1}."
       .format( best_result_rf['max_features'], best_result_rf['n_estimators'])))




    
