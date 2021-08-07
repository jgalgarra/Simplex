# README #

This repository contains the software, data and reproducibility instructions of a set of machine learning models to predict plant species abundance in Finca Caracoles, Spain.

Please, clone the repo and follow these steps.

## Data

There are two clean datasets called `abund_merged_dataset_onlycompetitors` and `abund_merged_dataset_onlyenvironment`. The first one records the abundance of competitor species whereas the second holds soil properties and annual precipitation.

## Exploratory analysis

You will need R 4.0.1 or later. Go to the Rscripts folder and run `exploratory.R`

- Results 

  * tables/abundance_distribution.csv   Abundance mean, median and variance by species
  * plots/EXPLO_individuals (both .png and. tiff) Abundance boxplots and Taylor's law fitting


## Machine Learning scripts

All machine learning scripts are written in Python and located at `Pyscripts` folder, you will need the release 3.8 or latter.

#### ALL_FEATURES_predict_by_year.py

This script builds three predictors (Linear regression, Random Forest and XGBoost) splitting data and trainig sets by YEAR.

Prediction errors (MSE, RMSE, RSE)  by model and year are stored at `results/ALLFEATURES_BY_YEAR.xlsx`

Invocation: `python ALL_FEATURES_predict_by_year.py`

If the invocation is `python ALL_FEATURES_predict_by_year n` instead,  then precipitation feature is excluded and results are stored at `results/ALLFEATURES_BY_YEAR_NOPRECIP.xlsx`

#### ABIOTIC_predictor.py

This script builds as many predictors as the number of experiments set by the contents of `experiments.txt`

Only abiotic features are included to train the models: linear regression, Random Forest and XGBoost. For each experiment, there is a random training/testing split.

Prediction errors (MSE, RMSE, RSE) by model and experiment are stored as individual sheets at 
`results/ABIOTIC_N.xlsx` where `N` stands for the number of experiments. 

Invocation: `python ABIOTIC_predictor.py`

If the invocation is `python ABIOTIC_predictor.py n` instead, then precipitation feature is excluded and the results
file is called `results/ABIOTIC_NOPRECIP_N.xlsx`

#### ALL_FEATURES_predictor.py

This script builds as many predictors as the number of experiments set by the contents of `experiments.txt`
All features are included to train the models: linear regression, Random Forest and XGBoost
For each experiment, there is a random training/testing split.

Prediction errors (MSE, RMSE, RSE) by model and experiment are stored as individual sheets at 
`results/ALLFEATURES_N.xlsx` where `N` stands for the number of experiments. 

Invocation: `python ALLFEATURES_predictor.py`

If the invocation is python `ALLFEATURES_predictor.py n`, then precipitation feature is excluded and the results
file is called `ALLFEATURES_NOPRECIP_N.xlsx`

#### TWO_STEP_predictor.py

This script builds as many predictors as the number of experiments set by the contents of `experiments.txt`
First, competitor species are predicted using abiotic features, using Random Forest. Then,
abiotic features and predicted competitors are joined to train the models: linear regression, Random Forest and XGBoost

Prediction errors (MSE, RMSE, RSE) by model and experiment are stored as individual sheets at 
`results/TWOSTEP_N.xlsx` where `N` stands for the number of experiments. 

Invocation: `python TWOSTEP_predictor.py`

CAUTION: This script is CPU-intensive. Running 100 experiments take hours.

#### TWO_STEP_predict_by_species.py

This script builds as many predictors as the number of experiments set by the contents of `experiments.txt`
First, competitor species are predicted using abiotic features, using Random Forest. Then,
abiotic features and predicted competitors are joined to train the models: linear regression, Random Forest and XGBoost

Prediction abundances by experiment are stored at `results/TWOSTEP_byspecies_N.xlsx` where `N` stands for the number of experiments. 

Invocation: `python TWO_STEP_predict_by_species.py`

CAUTION: This script is CPU-intensive. Running 100 experiments take hours.


## Feature importance scripts

Located at `Pyscripts/Feature_importance` folder. These scripts produce correlation matrices, importance tables and plots for the ABIOTIC and the ALL_FEATURES models. Results are stored in the same folder.

Invoke once: `python ABIOTIC_features_importance.py` and `ALL_FEATURES_features_importance.py

## Post-prediction scripts

A set of R scripts to produce plots and tables using the results of the predictors.
Invoke in the following order:

* `twostep_errors_species.R`
* `plot_results.R`
* `twostep_errors_species.R`
* `plot_error_byindividuals.R`
