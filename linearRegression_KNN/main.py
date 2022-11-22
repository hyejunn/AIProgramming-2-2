import numpy as np


def main():
    ml = buildModel()
    fileName = input("Enter the file name of training data: ")
    ml.setData('train', fileName)
    fileName = input("Enter the file name of test data: ")
    ml.setData('test', fileName)
    ml.testModel()
    ml.report()


def buildModel():
    print("Which learning algorithm do you want to use?")
    print(" 1. Linear Regression")
    print(" 2. k-NN")
    aType = int(input("Enter the number: "))
    if aType == 1:
        return LinearRegression()
    elif aType == 2:
        return KNN()


class ML:
    def __init__(self):
        self._trainDX = np.array([]) # Feature value matrix (training data)
        self._trainDy = np.array([]) # Target column (training data)
        self._testDX = np.array([])  # Feature value matrix (test data)
        self._testDy = np.array([])  # Target column (test data)
        self._testPy = np.array([])  # Predicted values for test data
        self._rmse = 0          # Root mean squared error

    def setData(self, dtype, fileName): # set class variables
        XArray, yArray = self.createMatrices(fileName)
        if dtype == 'train':
            self._trainDX = XArray
            self._trainDy = yArray
        elif dtype == 'test':
            self._testDX = XArray
            self._testDy = yArray
            self._testPy = np.zeros(np.size(yArray)) # Initialize to all 0
            
    def createMatrices(self, fileName): # Read data from file and make arrays
        infile = open(fileName, 'r')
        XSet = []
        ySet = []
        for line in infile:
            data = [float(x) for x in line.split(',')]
            features = data[0:-1]
            target = data[-1]
            XSet.append(features)
            ySet.append(target)
        infile.close()
        XArray = np.array(XSet)
        yArray = np.array(ySet)
        return XArray, yArray

    def testModel(self):
        pass

    def report(self):
        self.calcRMSE()
        print()
        print("RMSE: ", round(self._rmse, 2))

    def calcRMSE(self):
        n = np.size(self._testDy) # Number of test data
        totalSe = 0
        for i in range(n):
            se = (self._testDy[i] - self._testPy[i]) ** 2
            totalSe += se
        self._rmse = np.sqrt(totalSe) / n


class LinearRegression(ML):
    def __init__(self):
        ML.__init__(self)
        self._w = np.array([])  # Optimal weights for linear regression

    def linearRegression(self):  # Do linear regression and return optimal w
        X = self._trainDX
        n = np.size(self._trainDy)
        X0 = np.ones([n, 1])
        nX = np.hstack((X0, X))  # Add a column of all 1's as the first column
        y = self._trainDy
        t_nX = np.transpose(nX)
        return np.dot(np.dot(np.linalg.inv(np.dot(t_nX, nX)), t_nX), y)

    def testModel(self):
        self._w = self.linearRegression()
        n = np.size(self._testDy)
        self.testLR(n)

    def testLR(self, n):  # Test linear regression with the test set
        for i in range(n):
            self._testPy[i] = self.LR(self._testDX[i])

    def LR(self, data):  # Apply linear regression to a test data
        nData = np.insert(data, 0, 1)
        return np.inner(self._w, nData)


class KNN(ML):
    def __init__(self):
        ML.__init__(self)
        self._k = 0  # k value for k-NN

    def testModel(self):
        self._k = int(input("Enter the value for k: "))
        n = np.size(self._testDy)
        self.testKNN(n)

    def testKNN(self, n): # Apply k-NN to the test set
        for i in range(n):
            self._testPy[i] = self.kNN(self._testDX[i])

    ### Implement the following and other necessary methods
    def kNN(self, query):
        kNNIndex = self.findKNN(query)
        sumY = 0
        for i in range(self._k):
            sumY += self._trainDy[kNNIndex[i]]
        return sumY/self._k

    def findKNN(self, query):
        X = self._trainDX
        n = np.size(self._trainDy)
        distance = np.zeros(n)
        for i in range(n):
            distance[i] = np.linalg.norm(query-X[i])
        sortIndices = np.argsort(distance)
        kNNIndex = []
        for i in range(self._k):
            kNNIndex.append(sortIndices[i])
        return kNNIndex


main()
