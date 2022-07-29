
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

['Country name',
 'year',
 'Life Ladder',
 'Log GDP per capita',
 'Social support',
 'Healthy life expectancy at birth',
 'Freedom to make life choices',
 'Generosity',
 'Perceptions of corruption',
 'Positive affect',
 'Negative affect']
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

['Country Name',
 'Year',
 'Life Ladder',
 'Log Gdp Per Capita',
 'Social Support',
 'Healthy Life Expectancy At Birth',
 'Freedom To Make Life Choices',
 'Generosity',
 'Perceptions Of Corruption']
````
The column names are now capitalized. Next I need to check the column data types.
````
happiness.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1949 entries, 0 to 1948
Data columns (total 9 columns):
 #   Column                            Non-Null Count  Dtype  
---  ------                            --------------  -----  
 0   Country Name                      1949 non-null   object 
 1   Year                              1949 non-null   int64  
 2   Life Ladder                       1949 non-null   float64
 3   Log Gdp Per Capita                1913 non-null   float64
 4   Social Support                    1936 non-null   float64
 5   Healthy Life Expectancy At Birth  1894 non-null   float64
 6   Freedom To Make Life Choices      1917 non-null   float64
 7   Generosity                        1860 non-null   float64
 8   Perceptions Of Corruption         1839 non-null   float64
dtypes: float64(7), int64(1), object(1)
memory usage: 137.2+ KB
````
Here we see that 7 columns use data type float64. If I want the models to run, I need to convert the column values to the data type float32.
````
happiness.iloc[:, [2,3,4,5,6,7,8]] = happiness.iloc[:, [2,3,4,5,6,7,8]].astype(np.float32) 
happiness.info()
````
The models will also fail if there are missing values, so I will check for them.
````
print(happiness.isnull().sum()) #Find all rows with missing values
print(happiness.isnull().sum().sum()) #Total number of missing values

Country Name                          0
Year                                  0
Life Ladder                           0
Log Gdp Per Capita                   36
Social Support                       13
Healthy Life Expectancy At Birth     55
Freedom To Make Life Choices         32
Generosity                           89
Perceptions Of Corruption           110
dtype: int64
335
````
All the predictors have 335 total missing values. Removing them would drastically change the models. Instead, I will replace those values with their respective column means instead.
````
predictors = columns[3:9] #columns with NA values

for col in happiness[predictors]:
    happiness[col] = happiness[col].fillna(happiness[col].mean()) 

print(happiness.isnull().sum()) #should be 0

Country Name                        0
Year                                0
Life Ladder                         0
Log Gdp Per Capita                  0
Social Support                      0
Healthy Life Expectancy At Birth    0
Freedom To Make Life Choices        0
Generosity                          0
Perceptions Of Corruption           0
dtype: int64
````
There are no more missing values. The last thing I need to do is write the data to a CSV file so I can graph it in Tableau.
````
happiness.to_csv("happiness_cleaned.csv") 
````
## 2. Modeling
### 2a. Linear Regression

I need to extract my predictors and response variable from the dataframe. Then, I will split the dataset into a training set and a test set. I will also create a validation set and for all 3, compute the Mean Squared Error (MSE). The MSE measures the average squared difference between estimated values and actual values. I want to find the model that best minimizes it because it will best predict the happiness score.

````
#Life Ladder is what we are predicting so it will be excluded. Country Name and Year are not factors so they will also be dropped. 
X = happiness.drop(['Life Ladder', 'Country Name', 'Year'], axis = 1) 
Y = happiness['Life Ladder']
​
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .25, random_state = 15) #25% of the data will be tested
​
lr = LinearRegression().fit(X_train, y_train)
​
print(f"Training MSE : {mse(y_train, lr.predict(X_train))}") 
print(f"Test MSE     : {mse(y_test, lr.predict(X_test))}")
​
lr = LinearRegression()
print(f"CV MSE       : {-1 * cross_val_score(lr, X, Y, scoring = 'neg_mean_squared_error').mean()}")

Training MSE : 0.3018629550933838
Test MSE     : 0.39065566658973694
CV MSE       : 0.34414794743061067
````
The Linear Model's Test mean squared error is .391 and its Cross-validation mean squared error is .344.
![plot](/Graphs/Linear-Models.png)
I used Tableau to graph the linear models. Some social and economic variables like Log GDP Per Capita have a linear relationship with happiness score, while others like Generosity clearly do not.


### 2b. Decision Tree Model

The next model I will test is the Decision Tree Model. We will find the best split before testing the Model.

````
tree_reg = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)
tree_params = {'max_depth':np.linspace(1, 20, 20), 'max_features':np.linspace(0.1, 1, 10)}
tree_best = GridSearchCV(tree_reg, tree_params, cv=10, scoring = 'neg_mean_squared_error').fit(X_train, y_train)
tree_best.best_estimator_
DecisionTreeRegressor(max_depth=6.0, max_features=0.9, random_state=1)
````
The best split has a max depth of 6 and .9 max features.
````
tree_reg = DecisionTreeRegressor(max_depth = 6, max_features = 0.9, random_state = 1).fit(X_train, y_train)
````
The methods of calculating the mean squared errors are the same for all the tree based methods so I will write a function to avoid repetition. I will use 5 fold Cross-validation for all of them.
````
def tree_mse(model_reg): 
    print(f"Training MSE : {mse(y_train, model_reg.predict(X_train))}") 
    print(f"Test MSE     : {mse(y_test, model_reg.predict(X_test))}") 
​
    kf = KFold(n_splits=5, shuffle=True, random_state=35) #5 fold Cross-validation
​
    model_reg_scores = cross_val_score(estimator=model_reg, X=X, y=Y, cv=kf, scoring='neg_mean_squared_error')
    print(f"CV MSE       : {-np.mean(model_reg_scores)}")
````
````
tree_mse(tree_reg)

Training MSE : 0.19354378280424453
Test MSE     : 0.31653325082080946
CV MSE       : 0.31001827035029983
````
The Decision Tree Model's Test mean squared error is .317 and its Cross-validation mean squared error is .31.

Here is my Decision Tree.
````
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (30,20))
tree.plot_tree(tree_reg, feature_names = predictors, filled = True, fontsize=20)
plt.savefig('tree.png', dpi=300)
plt.show()

![plot](/Graphs/tree.png)

To create a scatterplot of my model's predictions in Tableau, I need to create a CSV file. I want to use this same CSV file and update it every time I make predictions with a new model. I will write a function that does this by adding predictions to a dataframe and then updating the CSV.
````
models = pd.DataFrame(data = dict({'Y-Test' : y_test})) #Use the dict() function to add the data

def add_predictions(name, values):
    models[name] = values
    models.to_csv("model_predictions.csv") #Create dataframe and then write it to CSV
    
add_predictions('Tree Predictions', tree_reg.predict(X_test))
````
![plot](/Graphs/Reg-Tree.png)
### 2c. Random Forest Model
````

````

````

````

````
### 2d. Gradient Boosting Model
````

````

````

````


````

````
### 2e. Bagging Model
````

````

````

````
#### 2f. Testing Performance Across Models
````

````

````

````
## Conclusions
````

````

````

````

````

````