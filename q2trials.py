import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats

data = pd.read_csv('continuous.csv', delimiter = ',')

#X = data[['# X1','X2','X3','X4','X5','X6','X7','X8','X9','X10']]
X = data[['X3','X7','X10']]
y = data['Y']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4)

#regr = LinearRegression()
#regr = Ridge(alpha = 1.0)
regr = Lasso(alpha = 1)
regr.fit(X_train, y_train)

coeff_df = pd.DataFrame(regr.coef_, X.columns, columns=['Coefficient'])

y_pred = regr.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

r2 = metrics.r2_score(y_test, y_pred)

print(coeff_df)
print(df)
print("r2 value: ", r2)


