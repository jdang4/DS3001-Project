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
from sklearn.linear_model import SGDClassifier


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

def define_for_price_vs_all(file) :
	x = file.iloc[:, 1:].values

	# standardizing all the x values
	x = preprocessing.StandardScaler().fit_transform(x)
	print(x)
	y = file.iloc[:, file.columns == 'SalePrice'].values

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

def compute_cost(b, dataFile) :
	cost = 0
	N, c = dataFile.shape

	for index, row in dataFile.iterrows():
		YearBuilt = coefficientMap.get('YearBuilt') * row['YearBuilt']
		YrSold = coefficientMap.get('YrSold') * row['YrSold']
		MonthSold = coefficientMap.get('MonthSold') * row['MonthSold']
		Size = coefficientMap.get('Size(sqf)') * row['Size(sqf)']
		Floor = coefficientMap.get('Floor') * row['Floor']
		TimeToSubway = coefficientMap.get('TimeToSubway') * row['TimeToSubway']
		TimeToBusStop = coefficientMap.get('TimeToBusStop') * row['TimeToBusStop']
		N_APT = coefficientMap.get('N_APT') * row['N_APT']
		N_manager = coefficientMap.get('N_manager') * row['N_manager']
		N_elevators = coefficientMap.get('N_elevators') * row['N_elevators']
		
		predicted_scores = YearBuilt + YrSold + MonthSold + Size + Floor + TimeToSubway + TimeToBusStop + N_APT + N_manager + N_elevators + b

		cost += pow((predicted_scores - row['SalePrice']), 2)

	average_of_squared_error = cost / (2 * N)

	return average_of_squared_error

#################################################################################

def step_gradient(b, dataFile, learning_rate) :
	w_gradient_yearBuilt = 0
	w_gradient_yearSold = 0
	w_gradient_monthSold = 0
	w_gradient_size = 0
	w_gradient_floor = 0
	w_gradient_subway = 0
	w_gradient_busStop = 0
	w_gradient_apt = 0
	w_gradient_manager = 0
	w_gradient_elevators = 0
	b_gradient = 0
	N, col = dataFile.shape

	for index, row in dataFile.iterrows():
		YearBuilt = coefficientMap.get('YearBuilt') * row['YearBuilt']
		YearSold = coefficientMap.get('YrSold') * row['YrSold']
		MonthSold = coefficientMap.get('MonthSold') * row['MonthSold']
		Size = coefficientMap.get('Size(sqf)') * row['Size(sqf)']
		Floor = coefficientMap.get('Floor') * row['Floor']
		TimeToSubway = coefficientMap.get('TimeToSubway') * row['TimeToSubway']
		TimeToBusStop = coefficientMap.get('TimeToBusStop') * row['TimeToBusStop']
		N_APT = coefficientMap.get('N_APT') * row['N_APT']
		N_manager = coefficientMap.get('N_manager') * row['N_manager']
		N_elevators = coefficientMap.get('N_elevators') * row['N_elevators']
		y = row['SalePrice']

		w_gradient_yearBuilt = (1/N) * row['YearBuilt'] * ((YearBuilt + b) - y)
		w_gradient_yearSold = (1/N) * row['YrSold'] * ((YearSold + b) - y)
		w_gradient_monthSold = (1/N) * row['MonthSold'] * ((MonthSold + b) - y)
		w_gradient_size = (1/N) * row['Size(sqf)'] * ((Size + b) - y)
		w_gradient_floor = (1/N) * row['Floor'] * ((Floor + b) - y)
		w_gradient_subway = (1/N) * row['TimeToSubway'] * ((TimeToSubway + b) - y)
		w_gradient_busStop = (1/N) * row['TimeToBusStop'] * ((TimeToBusStop + b) - y)
		w_gradient_apt = (1/N) * row['N_APT'] * ((N_APT + b) - y)
		w_gradient_manager = (1/N) * row['N_manager'] * ((N_manager + b) - y)
		w_gradient_elevators = (1/N) * row['N_elevators'] * ((N_elevators + b) - y)


	w_updated_yearBuilt = coefficientMap.get('YearBuilt') - (learning_rate * w_gradient_yearBuilt)
	coefficientMap['YearBuilt'] = w_updated_yearBuilt[0]

	w_updated_yearSold = coefficientMap.get('YrSold') - (learning_rate * w_gradient_yearSold)
	coefficientMap['YrSold'] = w_updated_yearSold[0]

	w_updated_monthSold = coefficientMap.get('MonthSold') - (learning_rate * w_gradient_monthSold)
	coefficientMap['MonthSold'] = w_updated_monthSold[0]

	w_updated_size = coefficientMap.get('Size(sqf)') - (learning_rate * w_gradient_size)
	coefficientMap['Size(sqf)'] = w_updated_size[0]

	w_updated_floor = coefficientMap.get('Floor') - (learning_rate * w_gradient_floor)
	coefficientMap['Floor'] = w_updated_floor[0]

	w_updated_subway = coefficientMap.get('TimeToSubway') - (learning_rate * w_gradient_subway)
	coefficientMap['TimeToSubway'] = w_updated_subway[0]

	w_updated_busStop = coefficientMap.get('TimeToBusStop') - (learning_rate * w_gradient_busStop)
	coefficientMap['TimeToBusStop'] = w_updated_busStop[0]

	w_updated_apt = coefficientMap.get('N_manager') - (learning_rate * w_gradient_apt)
	coefficientMap['N_manager'] = w_updated_apt[0]

	w_updated_manager = coefficientMap.get('N_manager') - (learning_rate * w_gradient_manager)
	coefficientMap['N_manager'] = w_updated_manager[0]

	w_updated_elevators = coefficientMap.get('N_elevators') - (learning_rate * w_gradient_elevators)
	coefficientMap['N_elevators'] = w_updated_elevators[0]


def gradient_descent_runner(b, learning_rate, numOfIterations, dataFile) :
	cost_list = [] #store cost in each iteration
	for _ in range(0, numOfIterations, 1) :
		step_gradient(b, dataFile, learning_rate)

		cost_list.append(compute_cost(b, dataFile))

	return cost_list


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


	"""
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
	"""
	################################################################################################################################

	# Fit Regression Model

	reducedDataFile = dataFile.copy(deep=True)
	reducedDataFile.drop(['N_Parkinglot(Ground)', 'N_Parkinglot(Basement)'], axis=1, inplace=True)

	#x, y = define_for_price_vs_all(reducedDataFile)
	x = reducedDataFile.iloc[:, 1:].values

	y = reducedDataFile.iloc[:, reducedDataFile.columns == 'SalePrice'].values
	# performing K-folds Cross Validation to minimize overfitting error
	# training: 70%
	# testing: 30%
	trainingX, testingX, trainingY, testingY = train_test_split(
		x, y, train_size =0.7, test_size = 0.3, random_state=0)

	lr_model = linear_model.LinearRegression(n_jobs=1)
	
	lr_model.fit(x, y)

	createCoefficientMap(reducedDataFile, lr_model)

	print(coefficientMap.get('Size(sqf)'))
	cost_list = gradient_descent_runner(lr_model.intercept_, 0.01, 1, reducedDataFile)
	
	print(coefficientMap.get('Size(sqf)'))

	iterations = [i for i in range(1)]

	#plt.scatter(iterations, cost_list)
	#plt.show()
	
	

	
	newX = reducedDataFile.iloc[:, 1:].values
	newY = reducedDataFile.iloc[:, reducedDataFile.columns == 'SalePrice'].values

	X_train, X_test, Y_train, Y_test = train_test_split(newX, newY, test_size=0.3, random_state=0)

	params = {
		'n_estimators' : 500,
		'max_depth' : 4,
		'min_samples_split' : 2,
		'learning_rate' : 0.01,
		'loss' : 'ls'
	}

	clf = ensemble.GradientBoostingRegressor(**params)

	clf.fit(X_train, Y_train.ravel())


	print(Y_test[0])

	#print(X_test[:1])

	value = [[2006, 2010, 7, 910, 1, 1, 2, 3, 3, 0]]
	#value = preprocessing.StandardScaler().fit(value)

	#print(value)
	#print(clf.predict(X_test[:1]))
	print(clf.predict(value))

	#print(clf.predict(value))
	new_rmse = mean_squared_error(Y_test, clf.predict(X_test))**0.5

	print('Root Mean Square Error: %.2f' % new_rmse)
	print(clf.score(X_train, Y_train))
	#plotFeatureImportance(clf, reducedDataFile)

	
