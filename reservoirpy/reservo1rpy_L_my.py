from reservoirpy.nodes import Reservoir, Input, Output
from reservoirpy.datasets import lorenz96
import matplotlib.pyplot as plt

## L
import numpy as np
from scipy.integrate import odeint

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
dt = 0.01

def f(state, t): #функция
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

x_train_length = 60.0 #<-- здесь выбираем, от 0 до каких пор знаем реальные значения, 6000 значений

state0 = [1.0, 1.0, 1.0]
time_steps = np.arange(0.0, x_train_length, dt) #<-- здесь выбираем, от 0 до каких пор знаем реальные значения

X = odeint(f, state0, time_steps) 
## L

##1
import reservoirpy as rpy

rpy.verbosity(0)
rpy.set_seed(42)  # сделать все воспроизводимым

reservoir1 = Reservoir(100, lr=0.5, sr=0.9) ##ГИПЕРПАРАМЕТРЫ

import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge, FORCE, ESN

##<-----здесь
##14

#readout = Ridge(ridge=1e-7)
##ГИПЕРПАРАМЕТР
#Параметр считывания на 1e-7. Это регуляризация, гиперпараметр, который поможет избежать переобучения.

##15
#Определите тренировочную задачу
#Подобные узлы Ridge можно обучать с помощью их fit() метода
#два временных ряда: входной временной ряд и целевой временной ряд.
X_train = X[:3000]
Y_train = X[1:3001]

##16
train_states = reservoir1.run(X_train, reset=True)
#reservoir1.run(X_train, reset=True)
##17
#тренируем
#warmup параметр - установить количество временных шагов, которые мы хотим отбросить
##ГИПЕРПАРАМЕТР
#readout = readout.fit(train_states, Y_train, warmup=10)


##19
#Создайте модель ESN
#ESN — это очень простой тип модели, содержащий два узла: резервуар и вывод.
#Чтобы объявить связи между узлами и построить модель, используйте >> оператор:

reservoir1 = Reservoir(100, lr=0.5, sr=0.9)
ridge1 = Ridge(ridge=1e-7)
#ridge1 = ridge1.fit(train_states, Y_train, warmup=10)
esn_model = reservoir1 >> ridge1

##20
#Тренируйте ESN
esn_model = esn_model.fit(X_train, Y_train, warmup=10)

##21
#print(reservoir1.is_initialized, readout.is_initialized, readout.fitted)

##22
#Запустите ESN
Y_pred = esn_model.run(X[3000:])

def plot_dimension(dim, name):
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    ax.plot(time_steps[3000:], X[3000:][:, dim], color = "black") #реал.
    ax.plot(time_steps[3000:], Y_pred[:, dim], "--", color = "gray") #предск.
    plt.xlabel("time")
    plt.ylabel(name) 
    plt.draw()
    plt.show()

plot_dimension(0, 'x')
plot_dimension(1, 'y')
plot_dimension(2, 'z')

plt.figure(figsize=(6, 4))
plt.plot(X[3000:][:, 0], X[3000:][:, 2], label="Ground truth", color = "black")
plt.plot(Y_pred[:, 0], Y_pred[:, 2],'--', label="Reservoir computing estimate", color = "gray")
plt.plot(X[3000:][0, 0], X[3000:][0, 2], "ko", label="Initial condition", markersize=8)

plt.legend()
plt.show()












