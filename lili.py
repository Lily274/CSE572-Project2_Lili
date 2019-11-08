# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import types
# from sklearn.decomposition import PCA
from scipy.fftpack import fft
from math import sqrt

plt.rcParams["figure.figsize"] = (20,10)
import sys

# file path
path = "./MealNoMealData/"

list_shapes_Meal = []
list_shapes_NoMeal = []

list_file_subjects = ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv"]
filename = ["mealData", "Nomeal"]

# load data
MealNoMealData = {}
for name in filename:
    MealNoMealData[name] = None
    for file_subject in list_file_subjects:
        df = pd.read_csv(path + name + file_subject, delimiter=',', names = list(range(31)))
        if MealNoMealData[name] is None:
            MealNoMealData[name] = df
        else:
            MealNoMealData[name] = pd.concat([MealNoMealData[name],df], axis = 0)
#print(MealNoMealData["mealData"])
#print(len(MealNoMealData["mealData"]))

# interpolation:
for name in filename:
    MealNoMealData[name].interpolate(axis=1,limit=60,limit_direction='both')

A = MealNoMealData["mealData"]
#print(MealNoMealData["mealData"].iloc[0])

# Extract features:
n = 8
Feature2 = {}
for name in filename:
    Feature2[name] = None
    i = 0
    while i<len(MealNoMealData[name]):
        tempt = MealNoMealData[name].iloc[i].values
        print("Data", tempt, "\n")
        i = i+1
        tempt_fft = abs(fft(tempt))/len(tempt)
        T = tempt_fft[range(1,n+1)]/sum(tempt_fft[range(1,n+1)])
        print("FFT", T, "\n")
        Feature2[name] = [Feature2[name],T]
print(Feature2)
        
"""
# apply interpolation
for i in range(len(list_file_subjects)):
    list_Meal[i] = list_Meal[i].interpolate(axis=1, limit=60, limit_direction='both')
    list_NoMeal[i] = list_NoMeal[i].interpolate(axis=1, limit=60, limit_direction='both')

#************ Feature 1 ***************
list_Meal_Feature1 = []
list_NoMeal_Feature1 = []

#************ Feature 2 ***************
list_Meal_Feature2 = []
list_NoMeal_Feature2 = []
n = 8
for i in range(len(list_file_subjects)):
    list_Meal_Feature2.append(pd.DataFrame(columns=["FFT 1","2","3","4","5","6","7","8"]))
    list_NoMeal_Feature2.append(pd.DataFrame(columns=["FFT 1","2","3","4","5","6","7","8"]))
for i in range(len(list_file_subjects)):
    tempt1 = list_Meal[i]
    tempt2 = list_NoMeal[i]
    
    for j1 in range(list_shapes_Meal[i][0]):
        tempt_event = tempt1.loc[j1].values
        tempt_fft = abs(fft(tempt_event))/len(tempt_event)
        if pd.isnull(tempt_fft).any():
            T = [0] * 8
        else:
            T = tempt_fft[range(1,n+1)]/sum(tempt_fft[range(1,n+1)])
        list_Meal_Feature2[i] = list_Meal_Feature2[i].append(pd.Series((T),index=list_Meal_Feature2[i].columns),
                                                             ignore_index=True)

    for j2 in range(list_shapes_NoMeal[i][0]):
        tempt_event = tempt2.loc[j2].values
        tempt_fft = abs(fft(tempt_event))/len(tempt_event)
        if pd.isnull(tempt_fft).any():
            T = [0] * 8
        else:
            T = tempt_fft[range(1,n+1)]/sum(tempt_fft[range(1,n+1)])
        list_NoMeal_Feature2[i] = list_NoMeal_Feature2[i].append(pd.Series((T),index=list_NoMeal_Feature2[i].columns),
                                                                 ignore_index=True)

print('Sample of Feature 2 for Meal: \n==============\n')
print(list_Meal_Feature2[0],'\n\n')
print('Sample of Feature 2 for NoMeal: \n==============\n')
print(list_NoMeal_Feature2[0],'\n\n')


#************** Feature 3 *****************
list_Meal_Feature3 = []
list_NoMeal_Feature3 = []

#************** Feature 4 *****************
list_Meal_Feature4 = []
list_NoMeal_Feature4 = []

list_Meal_Features = []
list_NoMeal_Features = []



# Test*****
list_Meal_Features = list_Meal_Feature2
list_NoMeal_Features = list_NoMeal_Feature2


for i in range(len(list_file_subjects)):
    print(np.array(list_Meal_Features[i].values))
  
    df_Meal_Features = pd.concat(
        [list_Meal_Feature1[i],list_Meal_Feature2[i],list_Meal_Feature3[i],list_Meal_Feature4[i]],
        axis=1).fillna(0)
    list_Meal_Features.append(df_Meal_Features)
    
    df_NoMeal_Features = pd.concat(
        [list_NoMeal_Feature1[i],list_NoMeal_Feature2[i],list_NoMeal_Feature3[i],list_NoMeal_Feature4[i]],
        axis=1).fillna(0)



# PCA
def zeroMean(dataMat):
    # 对每一列零均值化
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal

def pca(dataMat,n):
    # pca calculation, n is usually 5 in our project.
    newData,mealVal = zeroMean(dataMat)
    covMat = np.cov(newData,rowvar=0)

    eigVals,eigVects = np.linalg.eig(np.mat(covMat)) # get eigen values & eigen vectors
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]
    #print(n_eigValIndice)
    n_eigVect = eigVects[:,n_eigValIndice]
    return n_eigVect # return top 5 feature vectors.
    
# PCA processes the original data
array_Meal
newData_Meal,meanVal_Meal = zeroMean(np.array(list_Meal_Features[0].values))
print(newData_Meal,meanVal_Meal)
eigVect_Meal = pca(np.array(list_Meal_Features[0].values),5)
newData_NoMeal,meanVal_NoMeal = zeroMean(np.array(list_NoMeal_Features[0].values))

pca_Meal_Features = newData_Meal*eigVect_Meal*eigVect_Meal.T+meanVal_Meal
pca_NoMeal_Features = newData_NoMeal*eigVect_Meal*eigVect_Meal.T+meanVal_NoMeal


# KNN
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row,dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# KNN
def knn(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction
"""
