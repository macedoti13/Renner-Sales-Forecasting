# Renner's Weekly Sales Forecasting with Machine Learning
### Using machine learning to predict the weekly sales for each of Renner's stores.

## Table of Contents 
1. [Project Overview](#overview)
2. [The Problem that Needed to be Solved](#problem)
3. [File Descriptions](#files)
4. [Used Frameworks](#frameworks)
5. [Acknowledgements](#acknowledgements)
6. [Instructions](#instructions)

## Project Overview <a name="overview"><a/>
**The project is composed of building a machine learning model in order to predict the next 8 weeks sales for each of Renner's stores. The data that was given to us didn't have many columns, so, in order to get a good model, a complex ETL pipeline and Feature Engeneering processes were necessary. The steps in the project development were:**
1. Creating an ETL Pipeline that reads the data, clean and saves it into a csv file
2. Creating an Machine Learning Pipeline that trains a regression model in order the predict the number of sales of a product

## The Problem that Needed to be Solved <a name="problem"><a/>
**Predicting how many sales of a product are going to happen in each store of your business is extremelly useful. That way a company can produce just the right amount of each product and that, cut off unnecessary costs in production. With that in mind, Renner, one of the biggest clothing companies in Latim America, gave me a dataset with the weekly sales of one of their products in more than 400 stores. My objective was to build a machine learning model to forecast the sales for the next two months.**

## File Descriptions <a name="files"></a>
1. data
    - training_data.csv: csv file that contains the data used to train the model
    - testing_data.csv: csv file that contains the data used to validate the model
    - forecasted_data.csv: csv file that contains the forecasted data by the model
2. cleaned_data
    - cleaned_training_data.csv: csv file that contains the training data after being submitted to the etl pipeline
    - cleaned_testing_data.csv: csv file that contains the testing data after being submitted to the etl pipeline
3. models
    - not_tuned_model.pkl: File with the saved xgboost model (that wasn't tunded)
    - xgb_reg.pkl: File with the tuned xgboost model saved
4. research-notebooks
    - All the notebooks that were used for the project development (not necessary)
5. dataProcessing
    - Contains the DataProcessor class. This object cleans and performes feature engineering in the dataset and it's used in the etl pipeline
6. functions.py
    - Contians all the main functions that were utilized by the ETL and the ML pipelines
7. process_data.py
    - The ETL pipeline that reads the csv, cleans it, perform feature engineering and saves it into the 'cleaned_data' folder
8. train_forecaster.py
    - The ML pipeline that trains and evaluates an XGBoost Regression model and saves it to a pickle file.

## Used Framewors <a name="frameworks"></a>
- Pandas
- NumPy
- SKlearn
- XGBoost
- sys
- pickle

## Acknowledgements <a name="acknowledgements"></a>
- Time Series Forecasting with XGBoost from TDS: https://towardsdatascience.com/multi-step-time-series-forecasting-with-xgboost-65d6820bec39
- XGBoost hyperparameter tuning: https://www.anyscale.com/blog/how-to-tune-hyperparameters-on-xgboost
- Time Series Forecasting Tutorial: https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost

## Instructions <a name="instructions"></a>
- 1. Run the ETL Pipeline: `python3 process_data.py data/training_data.csv data/testing_data.csv`
- 2. Run the Machine Learning Pipeline: `python3 train_forecaster.py cleaned_data/cleaned_training_data.csv cleaned_data/cleaned_testing_data.csv data/testing.csv models/tuned_model`