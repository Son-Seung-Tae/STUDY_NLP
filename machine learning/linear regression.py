# keras로 구현하는 linear regression
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers

class LinearRegressionClass():

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(1, input_dim=1, activation="linear"))

    def loss_function(self, l_r = 0.01,):
        sgd = optimizers.SGD(lr= l_r)
        self.model.compile(optimizer=sgd, loss="mse", metrics=["mse"])

    def model_fit(self, x, y, epoches = 100):
        self.model.fit(x,y,epochs = epoches)

x = [1, 2, 3, 4, 5, 6, 7, 8, 9] # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95] # 각 공부하는 시간에 맵핑되는 성적

    
linear_r = LinearRegressionClass()
linear_r.loss_function()
linear_r.model_fit(x,y,epoches=300)
# %%
plt.plot(x, linear_r.model.predict(x))
plt.show()
