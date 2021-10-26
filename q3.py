import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import csv

data = pd.read_csv('pca_ex.csv')
X = data[['X1','X2','X3']]
#print(X)

with open('classes.txt', 'r') as txtfile:

    data = txtfile.read()
    data.split()

    Y = []
    
    for val in data.split():
        Y.append(int(val))


pca = PCA(n_components = 2)

principalComponents = np.array(pca.fit_transform(X))

principal1 = principalComponents[:,0]
principal2 = principalComponents[:,1]

print(principalComponents)
print(principal1)
print(principal2)

plt.scatter(principal1, principal2, c=Y)
plt.title("Principal component 1 vs Principal component 2")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.legend()

plt.show()
