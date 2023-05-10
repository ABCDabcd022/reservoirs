import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import sparse

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
dt = 0.01

def f(state, t): #функция
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

x_train_length = 40.0 #<-- здесь выбираем, от 0 до каких пор знаем реальные значения
sequence_length = 4000  #<-- здесь выбираем, от 0 до каких пор тренируемся
x_sim_length = int(x_train_length/dt) #<-- здесь выбираем, от 0 до каких пор предсказываем

state0 = [1.0, 1.0, 1.0]
time_steps = np.arange(0.0, x_train_length, dt) #<-- здесь выбираем, от 0 до каких пор знаем реальные значения


x_train = odeint(f, state0, time_steps) 

radius = 0.8
sparsity = 0.01
input_dim = 3
reservoir_size = 1000
n_steps_prerun = 10
regularization = 1e-2
sequence = []
for i in range(sequence_length): #<-- здесь выбираем, от 0 до каких пор тренируемся
    sequence.append(x_train[i])

#Случайным образом инициализируйте веса сети:
weights_hidden = sparse.random(reservoir_size, reservoir_size, density=sparsity)
eigenvalues, _ = sparse.linalg.eigs(weights_hidden)
weights_hidden = weights_hidden / np.max(np.abs(eigenvalues)) * radius

weights_input = np.zeros((reservoir_size, input_dim))
q = int(reservoir_size / input_dim)
for i in range(0, input_dim):
    weights_input[i * q:(i + 1) * q, i] = 2 * np.random.rand(q) - 1 #что за ":"?
    
#запись    
inp = weights_input
file = open('input.txt', 'w')
for i in range(len(inp)):
    file.write(str(inp[i][0])+'\n')
    file.write(str(inp[i][1])+'\n')
    file.write(str(inp[i][2])+'\n')
file.close()   


#чтение
inp1 = np.zeros((reservoir_size, input_dim))

file = open('input.txt', 'r')
j = 0
k = 0
for i in file:
    inp1[j][k % 3] = float(i)
    print(inp1[j][j % 3])
    print(j)
    k = k + 1
    if (k % 3 == 0):
        j = j + 1
file.close()    






















