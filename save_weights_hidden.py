import numpy as np
import pysindy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import sparse
from scipy.sparse import coo_matrix
 
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

#запись

row = weights_hidden.row
file = open('row.txt', 'w')
for i in range(len(row)):
    file.write(str(row[i])+'\n')
file.close()    
    
col = weights_hidden.col
file = open('col.txt', 'w')
for i in range(len(col)):
    file.write(str(col[i])+'\n')
file.close()   

print(col)

data = weights_hidden.data
file = open('data.txt', 'w')
for i in range(len(data)):
    file.write(str(data[i])+'\n')
file.close()   

shape = weights_hidden.shape
file = open('shape.txt', 'w')
for i in range(len(shape)):
    file.write(str(shape[i])+'\n')
file.close()   

print(1)
print(row)
print(col)
print(data)
print(shape)

#чтение

file = open('row.txt', 'r')
j = 0
for i in file:
    row[j] = float(i)
    j = j + 1
file.close()    
    

file = open('col.txt', 'r')
j = 0
for i in file:
    col[j] = float(i)
    j = j + 1
file.close()   


file = open('data.txt', 'r')
j = 0
for i in file:
    data[j] = float(i)
    j = j + 1
file.close()   

shape1 = []
file = open('shape.txt', 'r')
for i in file:
    shape1.append(int(i))
file.close()   

print(2)
print(row)
print(col)
print(data)
print(shape1)

#создание
weights_hidden1 = coo_matrix((data, (row, col)), shape=(shape1[0], shape1[1]))

















