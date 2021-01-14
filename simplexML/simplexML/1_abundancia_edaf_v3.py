"""
This script uses information on abiotic components to predict the number
of individuals that will appear in a certain subplot.

"""

import random

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import rse

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import xgboost


print("Predictor with environmental data only")
print("=======================================")

"""
Since we are going to work with abiotic components, just onte dataset will be used. 
There are some columns in the dataset that will not be used for prediction tasks. 
These are: year, month, day, plotID, x, y, subplot. 

"""

individuals_train = pd.read_csv('datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')

num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))

base_list = ['species','individuals','ph','salinity','cl','co3','c','mo','n','cn','p','ca','mg','k','na','precip']
col_list = base_list.copy()
col_list.append('present')
individuals_train = individuals_train[col_list]

individuals_types = individuals_train.dtypes

"Data Wrangling"

"Species feature is coded as numeric"
le = LabelEncoder()
le.fit(individuals_train[['species']])
individuals_train[['species']] = le.transform(individuals_train[['species']])

"Present feature is coded as numeric"
le = LabelEncoder()
le.fit(individuals_train[['present']])
individuals_train[['present']] = le.transform(individuals_train[['present']])

"""
Present feature indicates whether the number of individuals in that field is greater than 0 (True) or not.
There is 25% of the data in which present is True, that is, only in 25% of the rows the
number of individuals is> 0. It was proposed to do a SMOTE to balance the dataset, 
but the results worsened, what was discarded.

"""
    
"""
The reduction of dimensionality was also studied using a feature selection technique. 
It consists of introducing a random noise variable to later analyze which variables 
in my dataset provide more information than that random noise.

What was seen in this analysis is that there are only four variables that provide more 
information than random noise from the point of view of feature imporance, and they are: 
species, precip, present and co3. Carrying out the prediction with these variables 
did not provide better results, so the reduction in dimensionality was ruled out.

"""


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


"""
Since dimensionality reduction was ruled out, all variables will be used to train the model. 
The individuals variable is the one we want to predict, it is the number of individuals 
that will appear in a subplot, depending on the abiotic and biotic conditions.

For the train and test separation, in this case a random split is used 
(in another script some years are used as train and the last year as test).
"""

"Data Standarization"

variables_to_ignore = ['individuals']
selected_features = [element for element in list(individuals_train) if element not in variables_to_ignore]

individuals_model_train = individuals_train[selected_features]

std_scaler = StandardScaler()

std_scaler_model = std_scaler.fit(individuals_model_train)
individuals_model_train = std_scaler_model.transform(individuals_model_train)

"Train Test Split"

X = pd.DataFrame(data = individuals_model_train, columns = selected_features)
y = individuals_train.individuals


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8)
print(X_train.columns)

"Algorithms and Evaluation"

"Linear Regression"

reg = LinearRegression()
reg.fit(X_train,y_train)

predictions_lr = reg.predict(X_test)

rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, predictions_lr))
mse_lr = mean_squared_error(y_test,predictions_lr)
rse_lr = rse.calc_rse(y_test,predictions_lr)

print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_lr,rmse_lr,rse_lr))

"""
Finalmente se decidió utilizar como algoritmo random forest. Sin embargo, observamos que los resultados
en términos de rse no eran estables y diferían mucho de una ejecución a otra, por este motivo se decidió
utilizar validación cruzada.
"""

"Random Forest"

"Parámetros Random Forest"

# Number of trees in random forest
n_estimators = [100, 150]
# Number of features to consider at every split
max_features = ['auto']
#Grid Search
random_grid = {'n_estimators': n_estimators,
           'max_features': max_features}

print("Random Forest")
seed_value = 4
random.seed(seed_value)

regr = RandomForestRegressor( n_jobs = -1)
# regr = RandomForestRegressor(random_state= seed_value, n_jobs = -1, n_estimators = 150)
regr_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, cv = 7, verbose=2, n_jobs = -1)

regr_random.fit(X_train,y_train)
print(regr_random.best_params_)
predictions_rf = regr_random.best_estimator_.predict(X_test)

rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, predictions_rf))

mse_rf = mean_squared_error(y_test,predictions_rf)
rse_rf = rse.calc_rse(y_test,predictions_rf)

print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf,rmse_rf,rse_rf))


"XGBoost Regressor"


xgb = xgboost.XGBRegressor()
xgb.fit(X_train,y_train)

predictions_xgb = xgb.predict(X_test)

rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, predictions_xgb))
mse_xgb = mean_squared_error(y_test,predictions_xgb)
rse_xgb = rse.calc_rse(y_test,predictions_xgb)

print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_xgb,rmse_xgb,rse_xgb))
