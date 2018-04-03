import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def read(file):
	with open(file, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)

		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

def predict(dates, prices, x):
	dates = np.reshape(dates, (len(dates), 1))

	svr_linear = SVR(kernel='linear', C=1e3)
	svr_polynomial = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
	svr_linear.fit(dates, prices)
	svr_polynomial.fit(dates, prices)
	svr_rbf.fit(dates, prices)

	# plotting the initial datapoints 
	plt.scatter(dates, prices, color= 'black', label= 'Data')
	# plotting the line made by linear kernel
	plt.plot(dates,svr_linear.predict(dates), color= 'green', label= 'Linear model')
	# plotting the line made by polynomial kernel
	plt.plot(dates,svr_polynomial.predict(dates), color= 'blue', label= 'Polynomial model')
	# plotting the line made by the RBF kernel
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression(SVR)')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_linear.predict(x)[0], svr_polynomial.predict(x)[0]

if __name__ == "__main__":

	#plt.switch_backend('GTK')

	dates = []
	prices = []

	read('aapl.csv')

	predicted_price = predict(dates, prices, 29)  