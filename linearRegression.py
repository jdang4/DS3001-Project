import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import ensemble


def getCSV(path) :
	dataFile = pd.read_csv(path)
	return dataFile


TimeToSubway_quantitative = {
	'0-5min' : 4,
	'5min~10min' : 3,
	'10min~15min' : 2,
	'15min~20min' : 1,
	'no_bus_stop_nearby' : 0
}



TimeToBusStop_quantitative = {
	'0~5min' : 2,
	'5min~10min' : 1,
	'10min~15min' : 0
}

coefficientMap = {}

#################################################################################

def getCleanCSV(path, cleanPath) :
	read = pd.read_csv(path)

	read.sort_values(by='SalePrice').loc[read['SalePrice'] > 510000]

	with open(path, 'r') as file_in :
		with open(cleanPath, 'w') as file_out :
			writer = csv.writer(file_out)

			for row in csv.reader(file_in) :
				writer.writerow(row[:16])

	clean = pd.read_csv(cleanPath)

	return clean 

#################################################################################

def define_for_price_vs_all(dataFile) :
	x = dataFile.iloc[:, 1:].values

	# standardizing all the x values
	x = preprocessing.StandardScaler().fit_transform(x)

	y = dataFile.iloc[:, dataFile.columns == 'SalePrice'].values

	return x, y


#################################################################################

def plotLinearRegressionModel(lr_model, testingX, testingY) :
	plt.scatter(lr_model.predict(testingX), testingY)
	plt.xlabel('Prediction Price')
	plt.ylabel('Actual Price')

	plt.show()


#################################################################################

def showCoefficientsRegressionModel(coef, dataFile) :

	coef_df = pd.Series(coef[0], index=dataFile.columns[1:])

	plt.rcParams['figure.figsize'] = (8, 10)

	coef_df.plot(kind = 'barh')

	plt.title('Coefficients Regression Model')
	plt.xlabel('Coefficient Value')
	plt.show()

#################################################################################

def getBoxPlot(dataFile) :

	plt.rcParams['figure.figsize'] = (10, 10)
	plt.boxplot(x=dataFile['SalePrice'])
	plt.show()

#################################################################################

def createCoefficientMap(dataFile, lr_model) :
	tempList = lr_model.coef_

	coefficient_list = []
	for c in tempList[0] :
		coefficient_list.append(c)

	

	for i in range(1, len(dataFile.columns)) :
		coef_index = i - 1

		column_name = dataFile.columns[i]

		coefficientMap[column_name] = coefficient_list[coef_index]

#################################################################################

if __name__ == "__main__" :

	#dataFile = getCSV('dataset/Daegu_Real_Estate_data.csv')

	dataFile = getCleanCSV('dataset/Daegu_Real_Estate_data.csv', 
		'dataset/Daegu_Real_Estate_data_clean.csv')

	# mapping public transportation columns to its quantitative value (based on my own common sense)
	dataFile['TimeToSubway'] = dataFile['TimeToSubway'].map(TimeToSubway_quantitative)
	dataFile['TimeToBusStop'] = dataFile['TimeToBusStop'].map(TimeToBusStop_quantitative)

	# getting all the columns with quantitative types
	# referenced at https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html
	features = dataFile.dtypes[dataFile.dtypes != 'object'].index
	features = features[:13]
	dataFile = dataFile[features]

	
	x, y = define_for_price_vs_all(dataFile)


	# performing K-folds Cross Validation to minimize overfitting error
	# training: 70%
	# testing: 30%
	trainingX, testingX, trainingY, testingY = train_test_split(
		x, y, train_size =0.7, test_size = 0.3, random_state=0)

	lr_model = linear_model.LinearRegression(n_jobs=1)
	
	lr_model.fit(trainingX, trainingY)

	#plotLinearRegressionModel(lr_model, testingX, testingY)

	#showCoefficientsRegressionModel(lr_model.coef_, dataFile)
	
	createCoefficientMap(dataFile, lr_model)

	trainingY_prediction = lr_model.predict(trainingX)
	testingY_prediction = lr_model.predict(testingX)

	print('RMSE Train: %.2f' % mean_squared_error(trainingY, trainingY_prediction) ** 0.5)
	print('RMSE Test: %.2f' % mean_squared_error(testingY, testingY_prediction) ** 0.5)

	print('Root Mean Sqaure Error: %.2f' % (np.mean((lr_model.predict(testingX) - testingY)**2))**0.5)
	print('Variance Score: %.2f' % lr_model.score(testingX, testingY))

	
