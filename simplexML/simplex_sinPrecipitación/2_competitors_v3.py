
"""
En este script se utiliza información sobre componentes abióticos y componentes bióticos (número de individuos de otras especies
también presentes en el subplot) para predecir el número de individuals que aparecerán en un determinado subplot.

La diferencia con el modelo que se entrena en 1_abundacia_edaf_Comp es que en este caso no se parte de un dataset con todas las
variables y realizamos la predicción de individuals a partir de todas ellas. Lo que se hace en este script es un modelo predictivo en
dos pasos:
    1. Utilizamos los datos de componentes abióticos como features y la variable target a predecir será cada una de las diferentes especies de plantas
    competidoras.
    
    2. Una vez que se ha realizado la predicción del número de especies competidoras que crecerían en un determinado subplot a partir de
    esas condiciones abióticas, se utiliza como features (daaframe X) tanto esos parámetros edáficos como los componentes bióticos para predecir
    el número de individuals.

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
Dado que  se va a trabajar con componentes abióticos y bióticos, se cargarán ambos datasets y se realiza un merge de ambos  utilizando el índice como
columna de unión. Inicialmente en los dataset están presentes columnas que no se van a utilizar para
la predicción. Estas son: year, month, day, plotID, x, y, subplot. Además, existen columnas duplicadas en ambos datasets, por este motivo se 
realiza una selección de las variables que se utilizarán para entrenar el modelo.

Estas variables son las incluidas en col_list. Sin embargo, dado que el primer paso consiste en realizar la predicción del número de plantas de cada 
una de las especies que se desarrollarían por subplot en función de las condiciones abióticas, se ha generado train_list, que será el dataframe X.
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
       'k', 'na','present',
       'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']

train_list = ['species', 'individuals', 
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'present']


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

"""
La variable present indica si el número de individuals en ese terreno es mayor que 0 (True) o no.
Hay un 25% de los datos en los que present es True, es decir, que sólo en el 25% de las filas
el número de individuals es > 0. Se planteo hacer un SMOTE para balancear el dataset, pero los
resultados empeoraban, por lo que se descartó.

"""

"Estandarizacion de los datos"

conditions_model_train = conditions[train_list]

std_scaler = StandardScaler()
std_scaler_model = std_scaler.fit(conditions_model_train)
conditions_model_train = std_scaler_model.transform(conditions_model_train)



"""

Una vez que se han estandarizado los datos edáficos,se los asignamos al dataframe X. Dado que present sólo contiene los valores 0 y 1, no tiene sentido 
mantener los valores tras la estandarización por lo que se vuelven a reemplazar por 0 y 1.

La variable target y es un dataframe con todas las especies que se van a predecir. Para realizar las predicciones se ha impementado un bucle, de
forma que la variable a predecir será una especie diferente en cada iteración.

"""

"Division Train Test"

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


"Parámetros Random Forest"


n_estimators = [100,150]
max_features = ['auto']
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
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf,rmse_rf[variables_to_ignore],rse_rf))




"""

Una vez que ha concluido la predicción de cada una de las especies, se vuelve a redefinir el dataframe X con todas las variables abióticas y bióticas.
Para ello se utiliza la variable y_pred obtenida en el primer paso, que se une al X_test de componentes edáficos que nos habíamos dejado. Por
este motivo es necesario quitar el índice y volver a asignarlo.

Por otra parte,también necesitamos extraer aquellos registros del conjunto inicial para la variable individuals que coincidan con los datos 
asignados a test.

"""

"Utilizamos los resultados para predecir individuals"

features_to_pred = ['individuals']
selected_features = [element for element in col_list if element not in features_to_pred]

new_X = X_test_species.reset_index().drop(['index'], axis = 1)
y_predictions = pd.DataFrame.from_dict(y_pred)
y_predictions = y_predictions.applymap(lambda x: math.floor(x))
X_individuals = new_X.join(y_predictions)[selected_features]

y_individuals = conditions[features_to_pred].iloc[y_test_species.index].reset_index().drop(['index'], axis = 1)

data = X_individuals.join(y_individuals)

"""
Como el nuevo data corresponde al dataframe utilizado como test para el primer paso, se hace un remuestreo a partir de la varaible
present, que era la que estaba más desbalanceada.

"""
    
sm = SMOTE(random_state=42)
data, y_res = sm.fit_resample(data[['species', 'individuals',
    'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
    'k', 'na',
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

    
