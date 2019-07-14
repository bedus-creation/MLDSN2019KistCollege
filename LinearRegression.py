from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import numpy as np


class LinearRegression:
    def __init__(self, data):
        self.data = data

    def dealingNullValue(self):
        self.data = self.data.select_dtypes(
            include=[np.number]).interpolate().dropna()

    def labelData(self):
        y = np.log(self.data.SalePrice)
        X = self.data.drop(['SalePrice'], axis=1)
        return X, y

    def splitData(self, X, Y):
        return train_test_split(X, Y, random_state=42, test_size=.33)

    def fitModel(self, X_train, Y_train):
        lr = linear_model.LinearRegression()
        model = lr.fit(X_train, Y_train)
        return model

    def predict(self, model, X_test, Y_test):
        predictions = model.predict(X_test)
        return predictions

    def meanSquareError(self, y_test, predictions):
        print('RMSE is: \n', mean_squared_error(y_test, predictions))

    def showGraph(self, y_test, predictions):
        actual_values = y_test
        plt.scatter(predictions, y_test, alpha=.75, color='b')
        plt.xlabel('Predicted Price')
        plt.ylabel('Actual Price')
        plt.title('Linear Regression Model')
        plt.show()

    def filterCorrelated(self, index, CorrelationValue):
        Correlation = self.data.corr()
        cor_target = Correlation[index]
        relevant_features = cor_target[abs(cor_target) > CorrelationValue]
        Index = relevant_features.keys()
        self.data = pd.DataFrame(self.data, columns=Index)

    def run(self):
        # Dealing with null value
        self.dealingNullValue()
        # Lebel the data and target
        x, y = self. labelData()
        # split data into training and test datasets
        x_train, x_test, y_train, y_test = self.splitData(x, y)
        # Fit the model
        model = self.fitModel(x_train, y_train)
        # Calculate R2 score
        print("R^2 is: \n", model.score(x_test, y_test))
        # predict the model
        prediction = self.predict(model, x_test, y_test)
        # Mean Square Error
        self.meanSquareError(y_test, prediction)
        # show Graph
        self.showGraph(y_test, prediction)
