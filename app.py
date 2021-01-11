from sklearn import datasets
from SGDRegressor import SGD

X,y = datasets.load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=10)

#reg = LinearRegressionGD(epochs=10000,learning_rate=0.1)
sgd = SGD(epochs=10, learning_rate=0.1)

#reg.fit(X_train,y_train)
sgd.fit(X_train,y_train)

#y_pred = reg.predict(X_test)
y_pred1 = sgd.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred1))

