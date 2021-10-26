import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sys

#used to create a regression model of file1 and
#use it to predict file2's y values, and return an R^2 value.
def Regression(file1, file2):
    #load in data
    training_data = pd.read_csv(file1, delimiter = ',')
    testing_data = pd.read_csv(file2, delimiter = ',')

    #store data into training sets
    # (from the analysis, variables X1, X6, X4, X9, X2, X5, X8 were removed
    # in order as it was found they had little impact on the y values)
    X_train = training_data[['X3', 'X7', 'X10']]
    y_train = training_data['Y']

    #store data into testing sets
    X_test = testing_data[['X3', 'X7', 'X10']]
    y_test = testing_data['Y']

    #create linear regression model and fit it to training data
    regr = LinearRegression()
    regr.fit(X_train, y_train)

    #predict y values from testing data
    y_pred = regr.predict(X_test)

    #check r2 score (compare y values from test to real y values)
    r2 = metrics.r2_score(y_test, y_pred)

    print("Predicted Y values: ", y_pred)
    print("R^2 value: ", r2)

#used to split data into test/training sets and analyse it's success
#this function is not used in the main program
def Analysis(file1):
    data = pd.read_csv(file1, delimiter = ',')

    #store data
    X = data[['X3', 'X7', 'X10']]
    y = data['Y']

    #split into test/trainign sets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=3)
    
    #create and train model (ridge and lasso were found to be less effective)
    regr = LinearRegression()
    #regr = Ridge(alpha = 1.0)
    #regr = Lasso(alpha = 1.0)
    regr.fit(X_train, y_train)

    coeff_df = pd.DataFrame(regr.coef_, X.columns, columns=['Coefficient'])

    y_pred = regr.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    r2 = metrics.r2_score(y_test, y_pred)

    print(coeff_df)
    print(df)
    print("r2 value: ", r2)
    print(regr.summary())

#Analysis('continuous.csv')
Regression(sys.argv[1], sys.argv[2])
