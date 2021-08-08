"""
Created on Sun Aug  9 11:37:50 2020
@author: Iciar Civantos
This script builds computes the feature importance 'a posteriori' to compare
the RF and l inear regression models
"""

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

import sys
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

individuals_train = pd.read_csv('../../datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')

num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))

if include_precip:
   base_list = ['species','individuals','salinity','precip',
                'co3','c','p','ca']

else:
   base_list = ['species','individuals','salinity',
                'co3','c','p','ca']
col_list = base_list.copy()
individuals_train = individuals_train[col_list]
individuals_types = individuals_train.dtypes

"Data Wrangling"

le = LabelEncoder()
le.fit(individuals_train[['species']])
individuals_train[['species']] = le.transform(individuals_train[['species']])

if verbose:
    print(individuals_train.dtypes)
    

"Random Forest Feature Importance"

input_features = [column for column in individuals_train.columns if column != 'individuals']
X = individuals_train[input_features]
X['random_noise'] = np.random.normal(size=X.shape[0])
y = individuals_train['individuals']

# RF K-Fold train
regressor = RandomForestRegressor(n_jobs=-1)
cv = cross_validate(estimator=regressor, X=X, y=y, cv=5,return_estimator=True)

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

feature_importance.to_csv("feature_importance_ABIOTIC_rf_aposteriori.csv")


"Linear prediction summary"
import statsmodels.api as sm      
lm_1 = sm.OLS(y, X).fit()
print(lm_1.summary())
