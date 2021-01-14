
"""
This script uses information on abiotic components and biotic components (number of individuals of other species
also present in the subplot) to predict the number of individuals that will appear in a given subplot.


The difference with the model that is trained in 1_abundacia_edaf_Comp is that in this 
case we do not start from a dataset with all the variables and we make the prediction 
of individuals from all of them. What is done in this script is a predictive model 
in two steps:
     1. We use the data of abiotic components as features and the target variable 
     to predict will be each of the different species of competing plants.
    
     2. Once the prediction of the number of competing species that would grow in 
     a given subplot from these abiotic conditions has been made, both these edaphic 
     parameters and the biotic components are used as features (dataframe X) to 
     predict the number of individuals. .
"""

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import math

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
import xgboost
import rse

"""
Since we are going to work with abiotic and biotic components, both datasets will be 
loaded and both are merged using the index as a joining column. 
There are some columns in the datasets that will not be used for prediction tasks. 
These are: year, month, day, plotID, x, y, subplot. In addition, there are duplicate 
columns in both datasets. For this reason a selection is made of the variables that 
will be used to train the model.

These variables are the ones included in col_list. However, since the first step is 
to predict the number of plants of each of the species that would develop per 
subplot based on abiotic conditions, train_list has been generated, which will
 be the X dataframe.
"""

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

"Species feature is coded as numeric"
le = LabelEncoder()
le.fit(conditions[['species']])
conditions[['species']] = le.transform(conditions[['species']])


"Present feature is coded as numeric"
le = LabelEncoder()
le.fit(conditions[['present']])
conditions[['present']] = le.transform(conditions[['present']])

"""
Present feature indicates whether the number of individuals in that field is greater than 0 (True) or not.
There is 25% of the data in which present is True, that is, only in 25% of the rows the
number of individuals is> 0. It was proposed to do a SMOTE to balance the dataset, 
but the results worsened, what was discarded.

"""


"Standarization"

conditions_model_train = conditions[train_list]

std_scaler = StandardScaler()
std_scaler_model = std_scaler.fit(conditions_model_train)
conditions_model_train = std_scaler_model.transform(conditions_model_train)



"""

Once the edaphic data have been standardized, we assign them to the X dataframe.
 Since present only contains the values 0 and 1, it does not make sense to keep
 the values after standardization so they are replaced again by 0 and 1.

The variable target y is a dataframe with all the species to be predicted. 
To make the predictions, a loop has been implemented, so that the variable to 
be predicted will be a different species in each iteration.

"""

"Train Test Split"

X = pd.DataFrame(data = conditions_model_train, columns = train_list)
X[['present']] = conditions[['present']]


features_to_pred = ['BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']
y = conditions[features_to_pred]

X_train_species, X_test_species, y_train_species, y_test_species = train_test_split(X, y, train_size= 0.8)


y_pred = {}
rmse_rf = {}
rse_rf = {}


"Random Forest parameters"


n_estimators = [100,150]
max_features = ['auto']
random_grid = {'n_estimators': n_estimators,
           'max_features': max_features}



for i in range(0, len(features_to_pred)):

    variables_to_ignore = features_to_pred[i]
    print("--------------TARGET "+str(variables_to_ignore))
    
    "Train Test split"
    
    
    X_train = X_train_species
    y_train = y_train_species[variables_to_ignore]
    
    X_test = X_test_species
    y_test = y_test_species[variables_to_ignore]
    
        
    "Algorithms and evaluation"
    
    "Random Forest"
    
    rf = RandomForestRegressor(n_jobs = -1)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 7, verbose=2, n_jobs = -1)
    
    rf_random.fit(X_train,y_train)
    predictions_rf = rf_random.best_estimator_.predict(X_test)
    
    y_pred[variables_to_ignore] = predictions_rf
    rmse_rf[variables_to_ignore] = np.sqrt(metrics.mean_squared_error(y_test, predictions_rf))
    mse_rf = mean_squared_error(y_test,predictions_rf)
    rse_rf = rse.calc_rse(predictions_rf,mse_rf)
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf,rmse_rf[variables_to_ignore],rse_rf))




"""

Once the prediction for each of the species has been completed, the X dataframe 
is redefined with all the abiotic and biotic variables. For this, the variable 
y_pred obtained in the first step is used, which joins the X_test of edaphic 
components that we had left. For this reason, it is necessary to drop the 
index and reassign it.

On the other hand, we also need to extract those records from the initial set 
for the individuals variable that match the data assigned to test.

"""

"Results are used to predict individuals"

features_to_pred = ['individuals']
selected_features = [element for element in col_list if element not in features_to_pred]

new_X = X_test_species.reset_index().drop(['index'], axis = 1)
y_predictions = pd.DataFrame.from_dict(y_pred)
y_predictions = y_predictions.applymap(lambda x: math.floor(x))
X_individuals = new_X.join(y_predictions)[selected_features]

y_individuals = conditions[features_to_pred].iloc[y_test_species.index].reset_index().drop(['index'], axis = 1)

data = X_individuals.join(y_individuals)

"""
As the new data corresponds to the dataframe used as a test for the first step,
 a resampling is made from the variable present, which was the one that was the
 most unbalanced.

"""
    
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


"Algorithms and evaluation"

"Linear Regression"

reg = LinearRegression()
reg.fit(X_train_individuals,y_train_individuals)

predictions_lr = reg.predict(X_test_individuals)

rmse_lr = np.sqrt(metrics.mean_squared_error(y_test_individuals, predictions_lr))
mse_lr = mean_squared_error(y_test_individuals,predictions_lr)
rse_lr = rse.calc_rse(y_test_individuals,predictions_lr)

print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_lr,rmse_lr,rse_lr))

"Random Forest"
# print("Random Forest")
# seed_value = 4
# random.seed(seed_value)

# rf = RandomForestRegressor(random_state= seed_value, n_jobs = -1, n_estimators = 150)
# rf.fit(X_train_individuals,y_train_individuals)
# predictions_rf = rf.predict(X_test_individuals)

rf = RandomForestRegressor(n_jobs = -1)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 7, verbose=2, n_jobs = -1)

rf_random.fit(X_train_individuals,y_train_individuals)
predictions_rf = rf_random.best_estimator_.predict(X_test_individuals)

rmse_rf_final = np.sqrt(metrics.mean_squared_error(y_test_individuals, predictions_rf))
mse_rf_final = mean_squared_error(y_test_individuals,predictions_rf)
rse_rf_final = rse.calc_rse(y_test_individuals,predictions_rf)
print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf_final,rmse_rf_final,rse_rf_final))

"XGBoost Regressor"

xgb = xgboost.XGBRegressor()
xgb.fit(X_train_individuals,y_train_individuals)

predictions_xgb = xgb.predict(X_test_individuals)

rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test_individuals, predictions_xgb))
mse_xgb = mean_squared_error(y_test_individuals,predictions_xgb)
rse_xgb = rse.calc_rse(y_test_individuals,predictions_xgb)

print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_xgb,rmse_xgb,rse_xgb))

    
