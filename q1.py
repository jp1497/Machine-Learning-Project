# 4 binary random variables X1, X2, X3, Y
# write program that takes file as input and produces output
# maxmimum likelihood estimates and Bayesian estimate of the following
# probability distributions: P(Y), P(X1|Y), P(X2|Y), P(X3|Y)

import csv

with open('discrete.csv', 'r') as csvfile:
    next(csvfile)       #skip first row (column headers)
    readCSV = csv.reader(csvfile, delimiter = ',')
    
    X1 = []
    X2 = []
    X3 = []
    Y = []

    for row in readCSV:
        X1.append(int(row[0]))
        X2.append(int(row[1]))
        X3.append(int(row[2]))
        Y.append(int(row[3]))

# MLE: P = x / n
def MLE(Occurances,Total):
    return Occurances/Total

#counts number of occurances where Dataset1 = 0, Dataset2 = Condition
def CountOccurances(DataSet1,DataSet2,Condition):
    count = 0
    for index, val in enumerate(DataSet1):
        if DataSet1[index] == 0 and DataSet2[index] == Condition:
            count += 1
    return count

def CountY(Dataset,Condition):
    count = 0
    for val in Dataset:
        if val == Condition:
            count += 1
    return count

# uniform distribution == beta(1,1) -> this is our prior
# mean of beta distribution = a_posterior / a_posterior + b_posterior
# with a beta prior, we get a beta posterior
def MeanOfPosterior(Alpha_prior, Beta_prior, Occurances, Total):
    Alpha_posterior = Alpha_prior + Occurances
    Beta_posterior = Beta_prior + (Total - Occurances)
    return Alpha_posterior / (Alpha_posterior + Beta_posterior)

#function doesn't directly work for one dataset, but this is a workaround:
P1 = MLE(CountOccurances(Y, Y, 0), len(Y))
P2 = MLE(CountOccurances(X1, Y, 0), CountY(Y, 0))
P3 = MLE(CountOccurances(X1, Y, 1), CountY(Y, 1))
P4 = MLE(CountOccurances(X2, Y, 0), CountY(Y, 0))
P5 = MLE(CountOccurances(X2, Y, 1), CountY(Y, 1))
P6 = MLE(CountOccurances(X3, Y, 0), CountY(Y, 0))
P7 = MLE(CountOccurances(X3, Y, 1), CountY(Y, 1))

B1 = MeanOfPosterior(1, 1, CountOccurances(Y,Y,0), len(Y))
B2 = MeanOfPosterior(1, 1, CountOccurances(X1,Y,0), CountY(Y, 0))
B3 = MeanOfPosterior(1, 1, CountOccurances(X1,Y,1), CountY(Y, 1))
B4 = MeanOfPosterior(1, 1, CountOccurances(X2,Y,0), CountY(Y, 0))
B5 = MeanOfPosterior(1, 1, CountOccurances(X2,Y,1), CountY(Y, 1))
B6 = MeanOfPosterior(1, 1, CountOccurances(X3,Y,0), CountY(Y, 0))
B7 = MeanOfPosterior(1, 1, CountOccurances(X3,Y,1), CountY(Y, 1))
    
print("MLE:")
print("P(Y=0) = ", '%.3f' % P1)
print("P(X1=0|Y=0) = ", '%.3f' % P2)
print("P(X1=0|Y=1) = ", '%.3f' % P3)
print("P(X2=0|Y=0) = ", '%.3f' % P4)
print("P(X2=0|Y=1) = ", '%.3f' % P5)
print("P(X3=0|Y=0) = ", '%.3f' % P6)
print("P(X3=0|Y=1) = ", '%.3f' % P7)
print()    
print("Bayesian:")
print("P(Y=0) = ", '%.3f' %B1)
print("P(X1=0|Y=0) = ", '%.3f' % B2)
print("P(X1=0|Y=1) = ", '%.3f' % B3)
print("P(X2=0|Y=0) = ", '%.3f' % B4)
print("P(X2=0|Y=1) = ", '%.3f' % B5)
print("P(X3=0|Y=0) = ", '%.3f' % B6)
print("P(X3=0|Y=1) = ", '%.3f' % B7)


