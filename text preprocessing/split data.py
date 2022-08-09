import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sequences = [['a', 1], ['b', 2], ['c', 3]]
X, y = zip(*sequences)
print('X 데이터 :',X)
print('y 데이터 :',y)

X, y = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)

print(X_train)
print(y_train)

print(X_test)
print(y_test)