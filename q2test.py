import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats


data = pd.read_csv('continuous.csv', delimiter = ',')

X = data[['# X1', 'X2','X3', 'X4', 'X5', 'X6','X7', 'X8', 'X9', 'X10']]
y = data['Y']


X2 = sm.add_constant(X)
est = sm.OLS(y, X2)

est2 = est.fit()
print(X)
print(y)
