import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


def sigmoid(x):
    return 1/ (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

# sigmoid function figure
# plt.plot(x, y, 'g')
# plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
# plt.title('Sigmoid Function')
# plt.show()


x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# keras
model = Sequential()
model.add(Dense(1, input_dim=1, activation="sigmoid"))

sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["binary_accuracy"])

model.fit(x, y, epochs=500)

# plt.plot(x, model.predict(x), 'b', x,y, 'k.')
# plt.show()

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x = [[k]for k in x]
print(x)

y = [[k]for k in y]
print(y)

x_train = torch.FloatTensor(x)
y_train = torch.FloatTensor(y)

print(x_train)
print(y_train)

W = torch.zeros((1,1), requires_grad=True) # 크기는 2 x 1
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 500
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    # cost = -(y_train * torch.log(hypothesis) + 
    #          (1 - y_train) * torch.log(1 - hypothesis)).mean()
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
# print(W)
# print(b)

# hypothesis = torch.sigmoid(x_train.matmul(W) + b)

# F.binary_cross_entropy(hypothesis, y_train)

