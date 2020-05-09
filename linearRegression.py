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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


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

def plotFeatureImportance(clf, dataFile) :
	feature_importance = clf.feature_importances_

	feature_importance = 100.0 * (feature_importance / feature_importance.max())

	sorted_idx = np.argsort(feature_importance)

	dataFile_temp = dataFile

	dataFile_temp.drop('SalePrice', axis=1, inplace=True)
	pos = np.arange(sorted_idx.shape[0]) + 0.5

	plt.rcParams['figure.figsize'] = (15, 10)
	plt.subplot(1, 2, 2)
	plt.barh(pos, feature_importance[sorted_idx], align='center')
	plt.yticks(pos, dataFile_temp.columns[sorted_idx])
	plt.show()

#################################################################################

if __name__ == "__main__" :

	dataFile = getCleanCSV('dataset/Daegu_Real_Estate_data.csv', 
		'dataset/Daegu_Real_Estate_data_clean.csv')

	dataFile_original = dataFile.copy(deep=True)

	# mapping public transportation columns to its quantitative value (based on my own common sense)
	dataFile['TimeToSubway'] = dataFile['TimeToSubway'].map(TimeToSubway_quantitative)
	dataFile['TimeToBusStop'] = dataFile['TimeToBusStop'].map(TimeToBusStop_quantitative)

	# getting all the columns with quantitative types
	# referenced at https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html
	features = dataFile.dtypes[dataFile.dtypes != 'object'].index
	features = features[:13]
	dataFile = dataFile[features]

	x = dataFile.iloc[:, 1:].values

	y = dataFile.iloc[:, dataFile.columns == 'SalePrice'].values

	# performing K-folds Cross Validation to minimize overfitting error
	# training: 70%
	# testing: 30%
	trainingX, testingX, trainingY, testingY = train_test_split(
		x, y, train_size =0.7, test_size = 0.3, random_state=0)

	lr_model = linear_model.LinearRegression(n_jobs=1)
	
	lr_model.fit(trainingX, trainingY)

	#plotLinearRegressionModel(lr_model, testingX, testingY)

	#showCoefficientsRegressionModel(lr_model.coef_, dataFile)

	trainingY_prediction = lr_model.predict(trainingX)
	testingY_prediction = lr_model.predict(testingX)

	print('RMSE Train: %.2f' % mean_squared_error(trainingY, trainingY_prediction) ** 0.5)
	print('RMSE Test: %.2f' % mean_squared_error(testingY, testingY_prediction) ** 0.5)

	print('Root Mean Sqaure Error: %.2f' % (np.mean((lr_model.predict(testingX) - testingY)**2))**0.5)
	print('Variance Score: %.2f' % lr_model.score(testingX, testingY))
	
	################################################################################################################################

	# Fit Regression Model

	reducedDataFile = dataFile.copy(deep=True)
	reducedDataFile.drop(['N_Parkinglot(Ground)', 'N_Parkinglot(Basement)'], axis=1, inplace=True)

	newX = reducedDataFile.iloc[:, 1:].values
	newY = reducedDataFile.iloc[:, reducedDataFile.columns == 'SalePrice'].values

	X_train, X_test, Y_train, Y_test = train_test_split(newX, newY, test_size=0.3, random_state=0)

	params = {
		'n_estimators' : 500,
		'max_depth' : 20,
		'min_samples_split' : 2,
		'learning_rate' : 0.01,
		'loss' : 'ls',
		'random_state' : 0
	}

	clf = ensemble.GradientBoostingRegressor(**params)

	clf.fit(X_train, Y_train.ravel())

	value = [[2006, 2010, 7, 1910, 1, 1, 2, 3, 3, 0]]

	default_yrBuilt = reducedDataFile.mean()[1]
	default_yrSold = reducedDataFile.mean()[2]
	default_monthSold = reducedDataFile.mean()[3]
	default_size = reducedDataFile.mean()[4]
	default_floor = reducedDataFile.mean()[5]
	default_busStop = reducedDataFile.mean()[6]
	default_subway = reducedDataFile.mean()[7]
	default_apt = reducedDataFile.mean()[8]
	default_manager = reducedDataFile.mean()[9]
	default_elevator = reducedDataFile.mean()[10]

	print(default_yrBuilt)
	
	print('Predicted SalePrice: %.2f' % clf.predict(value)[0])

	new_rmse = mean_squared_error(Y_test, clf.predict(X_test))**0.5

	print('New Root Mean Square Error: %.2f' % new_rmse)
	print('New Variance Score:', clf.score(X_train, Y_train))
	#plotFeatureImportance(clf, reducedDataFile)


	################################################################################################################################

	# Decision Tree Classifier
	
	# performing K-folds Cross Validation to minimize overfitting error
	# training: 70%
	# testing: 30%
	trainingX, testingX, trainingY, testingY = train_test_split(
		x, y, train_size =0.7, test_size = 0.3, random_state=0)

	decisionTree = DecisionTreeClassifier(max_depth = 10, random_state=1)

	decisionTree.fit(trainingX, trainingY)
	
	y_prediction = decisionTree.predict(testingX)
	
	print('Decision Tree Classifier Accuracy:', accuracy_score(testingY, y_prediction))