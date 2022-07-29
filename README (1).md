
# What social and economic factors are most important in predicting a country's happiness?

The World Happiness Report is a publication that uses survey data to show how people across the world rate their happiness. 
An entry in the report is a country's average response to life evaluation questions for a given year between 2005 to 2020. Several social and economic variables are measured in these questions, such as the Freedom To Make Life Choices and Log GDP Per Capita. I was wondering which of these factors are the most important in predicting a country's happiness? I also want to find out which type of model performs the best at predicting a country's happiness. 

To answer these questions, I will test the Machine Learning Algorithms Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and Bagging on the World Happiness Report Dataset to find the the model that predicts the data best and identify the most statistically significant variables. I will graph the models using Tableau. 

## 1. Cleaning the data 

First I will load in the Python libraries I need like Pandas and download the dataset off the World Happiness Report website https://worldhappiness.report/ed/2021/#appendices-and-data.

    import os
    import pandas as pd
    import numpy as np 
    import random
    import matplotlib.pyplot as plt
    import seaborn as sns 
    import statsmodels.api as sm
    import mlxtend

    from sklearn import tree
    from sklearn.tree import export_graphviz, DecisionTreeRegressor
    from sklearn.feature_selection import SequentialFeatureSelector, RFE
    from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV #splits the data
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
    from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, scale
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV 
    from tabulate import tabulate
````
os.chdir(r"C:\Users\maazc\Desktop\Notes\Python\Data Project") #Navigate to the proper directory. The r is there before the " " to convert it to a raw string. \U is read as unicode
Happiness = pd.read_excel("DataPanelWHR2021C2.xls") #load in the data
happiness = Happiness.copy() #Our dataframe
````
Now I can clean the data starting with the variables.

````
list(happiness.columns.values)
````
Life Ladder or the Cantril Scale is a question where respondents are asked to rate their life on a scale from 0 through 10 where 0 is the worst possible life for them and 10 is the best possible life. This is the happiness score the models will try to predict.

Positive Affect and Negative Affect are other metrics of happiness that will be dropped to make the analysis simpler. But they are worth looking into another time.

Log GDP Per Capita, Social Support, Healthy Life Expectancy At Birth, Freedom To Make Life Choices, Generosity, and Perceptions Of Corruption are the social and economic factors that will predict happiness.

````
happiness = happiness.drop(['Positive affect', 'Negative affect'], axis = 1) #remove Positive affect and Negative affect
````
I want to capitalize all the column names.
````
happiness.columns = map(str.title, happiness.columns) #Capitalize the first letter of each column name
columns = list(happiness.columns.values) #column names
columns
````


