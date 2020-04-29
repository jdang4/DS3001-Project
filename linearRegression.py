import numpy as np 
import pandas as pd 
from collections import Counter
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt 


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


if __name__ == "__main__" :

	dataFile = getCSV('dataset/Daegu_Real_Estate_data.csv')

	# mapping public transportation columns to its quantitative value (based on my own common sense)
	dataFile['TimeToSubway'] = dataFile['TimeToSubway'].map(TimeToSubway_quantitative)
	dataFile['TimeToBusStop'] = dataFile['TimeToBusStop'].map(TimeToBusStop_quantitative)

	corr = dataFile.corr()

	print(corr.iloc[0])




