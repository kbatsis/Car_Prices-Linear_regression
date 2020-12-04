"""
Author: Kostas Batsis
Platform: Anaconda Python 3.8
"""

import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import missingno as msno

#Creates a single dataframe with all the relevant data
carData = pd.DataFrame()
dirEntries = os.listdir('cardata_kaggle')
for entry in dirEntries:
        currentFrame = pd.read_csv('cardata_kaggle/'+entry)
        currentFrame['Brand'] = entry.split('.')[0]
        carData = carData.append(currentFrame,ignore_index=True)
carData = carData.drop('tax',axis=1)
carData = carData.drop('tax(£)',axis=1)
carData = carData.drop('model',axis=1)
originalSampleSize = len(carData)

print(carData.isnull().sum()) #checks for missing data
msno.matrix(carData)
print('\n',carData[carData['mpg'].isnull()==True].fuelType.value_counts(),'\n')
carData = carData.drop('mpg',axis=1)

#Creates dummy variables for the nominal variables
brandCounts = carData['Brand'].value_counts()
dummy = pd.get_dummies(carData['Brand'],drop_first=True)
carData = pd.concat([carData,dummy],axis=1)
transmissionCounts = carData['transmission'].value_counts()
dummy = pd.get_dummies(carData['transmission'],drop_first=True)
carData = pd.concat([carData,dummy],axis=1)
fuelCounts = carData['fuelType'].value_counts()
dummy = pd.get_dummies(carData['fuelType'],drop_first=True)
carData = pd.concat([carData,dummy],axis=1)

#Plotting all the numerical variable distributions
plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(2,2,figsize=(27,15))
axs[0,0].hist(carData['price'],bins=20)
axs[0,0].set_title('Price (£)')
axs[0,1].hist(carData['year'],bins=20)
axs[0,1].set_title('Year')
axs[1,0].hist(carData['mileage'],bins=20)
axs[1,0].set_title('Mileage (miles)')
axs[1,1].hist(carData['engineSize'],bins=20)
axs[1,1].set_title('Engine Size (litres)')

#Dropping erroneous/illogical data and transforming the DV
carData = carData.drop(index=48528) #Case has a year value of 2060 
print('\n',carData[carData['engineSize']==0.0].fuelType.value_counts(),'\n')
carData = carData.drop(carData[carData['engineSize']==0.0].index,axis=0)
carData = carData.drop('Electric',axis=1)
carData['priceln'] = np.log(carData['price'])
plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(2,2,figsize=(27,15))
axs[0,0].hist(carData['year'],bins=20)
axs[0,0].set_title('Year')
axs[0,1].hist(carData['engineSize'],bins=20)
axs[0,1].set_title('Engine Size (litres)')
axs[1,0].hist(carData['priceln'])
axs[1,0].set_title('Log Price (£)')

#3D scatterplot for multivariate outliers among the IVs
fig = plt.figure(figsize=(17,13))
plt.rcParams.update({'font.size': 14})
ax = fig.add_subplot(111, projection='3d')
ax.scatter(carData['year'],carData['mileage'],carData['engineSize'])
ax.set_xlabel('Year')
ax.set_ylabel('mileage')
ax.set_zlabel('Engine size')

#Creates the dependent variables matrix and the independent var. vector
xSet = carData.drop(['price','priceln','transmission','Brand','fuelType'],axis=1)
ySet = carData['priceln']

#Examining multicollinearity through the VIF
vifData = pd.DataFrame()
vifData['Regressor'] = xSet.columns
vifData['VIF'] = [variance_inflation_factor(xSet.values, i) for i in range(len(xSet.columns))] 

#Fitting the regression equation
xSet = sm.add_constant(xSet)
carModel = sm.OLS(ySet,xSet).fit()
predictedValues = carModel.predict(xSet) 

fig = plt.figure()
plt.rcParams.update({'font.size': 8})
plt.scatter(predictedValues,predictedValues-ySet,label='Residual Plot') 
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')

outRes = xSet.loc[90588] #outlier in residual plot
xSet = xSet.drop(index=90588) 
ySet = ySet.drop(index=90588)

carModel = sm.OLS(ySet,xSet).fit()
predictedValues = carModel.predict(xSet) 

fig = plt.figure()
plt.rcParams.update({'font.size': 8})
plt.scatter(predictedValues,predictedValues-ySet,label='Residual Plot') 
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')

print(carModel.summary())