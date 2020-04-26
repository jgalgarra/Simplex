
import random

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet
import xgboost


environment_train = pd.read_csv('datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')
competitors_train = pd.read_csv('datasets/abund_merged_dataset_onlycompetitors.csv', sep=',')

individuals_train = environment_train.merge(competitors_train)
 

num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))


col_list = ['year', 'species', 'individuals', 'plotID', 'x', 'y',
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'plot', 'subplot', 'precip', 'sum_salinity', 'present',
       'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']

individuals_train = individuals_train[col_list]

individuals_types = individuals_train.dtypes

"Data Wrangling"

"Transformamos la variable species a numérica"
le = LabelEncoder()
le.fit(individuals_train[['species']])
individuals_train[['species']] = le.transform(individuals_train[['species']])

"Transformamos la variable plotID a numérica"
le = LabelEncoder()
le.fit(individuals_train[['plotID']])
individuals_train[['plotID']] = le.transform(individuals_train[['plotID']])

"Transformamos la variable subplot a numérica"
le = LabelEncoder()
le.fit(individuals_train[['subplot']])
individuals_train[['subplot']] = le.transform(individuals_train[['subplot']])

"Transformamos la variable present a numérica"
le = LabelEncoder()
le.fit(individuals_train[['present']])
individuals_train[['present']] = le.transform(individuals_train[['present']])

print(individuals_train.dtypes)

"Ver si hay registros duplicados"

num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)

individuals_train = individuals_train.drop_duplicates()
num_rows_clean = len(individuals_train)
print("In this dataset there were {} repeated records".format(num_rows - num_rows_clean))


"Estudiar qué variables toman siempre el mismo valor"

variables_numericas = [index for index, value in zip(individuals_train.dtypes.index, individuals_train.dtypes.values) 
                       if str(value) != 'object']

variances = individuals_train[variables_numericas].var(axis=0)
unique_values = []
for element in variables_numericas:
    unique_values.append(len(individuals_train[element].unique()))
    
data = []
for variable, variance, unique_values in zip(variables_numericas, variances, unique_values):
    data.append([variable, variance, unique_values])
    
variance_study = pd.DataFrame(data, columns=['variable', 'variance', 'unique_values']).sort_values(['unique_values','variance'], ascending=[True, True])
print(variance_study)

columns_to_delete = variance_study[variance_study.variance < 0.1].variable.tolist()

individuals_train.drop(columns_to_delete, axis=1, inplace=True)

"Estudio de valores nulos"

num_rows = len(individuals_train)
nulls_info = []

for column in list(individuals_train):
    
    number_nulls = len(individuals_train[pd.isnull(individuals_train[column])])    
    nulls_info.append([column,number_nulls*100/num_rows])    
    
nulls_info_df = pd.DataFrame(nulls_info, columns=['variable', 'percentage_nulls']).sort_values('percentage_nulls',ascending=False)
nulls_info_df

"Estudio de correlaciones"

variables_to_ignore = ['individuals']
variables_study = [element for element in list(individuals_train) if element not in variables_to_ignore]

correlation_matrix = individuals_train[variables_study].corr(method='spearman')

figure_size = (18, 14)
fig, ax = plt.subplots(figsize=figure_size)

sns.heatmap(correlation_matrix, xticklabels=list(correlation_matrix), yticklabels=list(correlation_matrix),
            annot=True, fmt='.1f', linewidths = 0.5, ax=ax)

num_variables = len(correlation_matrix)
correlation_value = 0.7
correlated_variables = []

for row in range(1,num_variables):
    for col in range(0,row):
        if abs(correlation_matrix.iloc[row,col]) >= correlation_value:
            print('The variable {0} is correlated with the variable {1} with a factor of {2}.'.format(list(correlation_matrix)[row], list(correlation_matrix)[col], correlation_matrix.iloc[row,col]))
            correlated_variables.append([list(correlation_matrix)[row], list(correlation_matrix)[col], correlation_matrix.iloc[row,col]])

correlated_variables = correlated_variables[1:]
correlated_variables = sorted(correlated_variables, key=lambda value: abs(value[-1]), reverse=True)
correlated_variables

"Correlacion con variable objetivo"

columns_to_delete_correlation_target = []
for element in correlated_variables:
    if element[0] not in list(individuals_train) or element[1] not in list(individuals_train):
        break
    else:
        first = individuals_train.individuals.corr(individuals_train[element[0]], method='spearman')
        second = individuals_train.individuals.corr(individuals_train[element[1]], method='spearman')
        if abs(first) > abs(second):
            print('The variable {0} is more correlated with the objective variable "individuals".'.format(element[0]))
            columns_to_delete_correlation_target.append(element[1])
        else: 
            print('The variable {0} is more correlated with the objective variable "individuals".'.format(element[1]))
            columns_to_delete_correlation_target.append(element[0])

columns_to_delete_correlation_target = list(set(columns_to_delete_correlation_target))
individuals_train.drop(columns_to_delete_correlation_target, axis=1, inplace=True)

num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)
print("This dataset has {0} records and {1} columns".format(num_rows, num_cols))

"Feature Importance"

input_features = [column for column in list(individuals_train) if column != 'individuals']
X = individuals_train[input_features]
X['random_noise'] = np.random.normal(size=X.shape[0])
y = individuals_train['individuals']

# RF K-Fold train
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


"Algoritmos y Evaluación"

"Random Forest"

seed_value = 4
random.seed(seed_value)
rf = RandomForestRegressor(random_state= seed_value)

param_grid = {'max_features':['auto', 'log2'], 'n_estimators':[100,150]}
cross_val_rf = GridSearchCV(rf, param_grid, cv = 5)
cross_val_rf.fit(X_train,y_train)
predictions_rf = cross_val_rf.predict(X_test)

rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, predictions_rf))

best_result_rf = cross_val_rf.best_params_

(print("The best random forest has a max_features value of {0} and n_estimators of {1}."
       .format( best_result_rf['max_features'], best_result_rf['n_estimators'])))


"Gradient Boosting Trees"

gbt = GradientBoostingRegressor(random_state= seed_value)

param_grid = {'learning_rate':[0.01,0.1], 'n_estimators':[150,250], 'max_depth':[7,8]}
cross_val_gbt = GridSearchCV(gbt, param_grid, cv = 5)
cross_val_gbt.fit(X_train,y_train)
predictions_gbt = cross_val_gbt.predict(X_test)

rmse_gbt = np.sqrt(metrics.mean_squared_error(y_test, predictions_gbt))

best_result_gbt = cross_val_gbt.best_params_

(print("The best gbt has a n_estimators value of {0}, max_depth of {1} and learning_rate of {2}."
       .format( best_result_gbt['n_estimators'], best_result_gbt['max_depth'], best_result_gbt['learning_rate'])))


"XGBoost"

xgb = xgboost.XGBRegressor(colsample_bytree=0.8,
                 gamma=0.1,                 
                 learning_rate=0.07,
                 max_depth=5,
                 min_child_weight=10,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 


xgb.fit(X_train,y_train)
predictions_xgb = xgb.predict(X_test)

rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, predictions_xgb))



