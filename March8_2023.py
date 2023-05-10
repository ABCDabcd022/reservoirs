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

#1
x_train = odeint(f, state0, time_steps) 

radius = 0.8
sparsity = 0.01
input_dim = 3
reservoir_size = 999
n_steps_prerun = 200
regularization = 1e-2
#2
sequence = []
for i in range(sequence_length): #<-- здесь выбираем, от 0 до каких пор тренируемся
    sequence.append(x_train[i])
#3
#Случайным образом инициализируйте веса сети:
weights_hidden = sparse.random(reservoir_size, reservoir_size, density=sparsity)
eigenvalues, _ = sparse.linalg.eigs(weights_hidden)
weights_hidden = weights_hidden / np.max(np.abs(eigenvalues)) * radius
#4
weights_input = np.zeros((reservoir_size, input_dim))
q = int(reservoir_size / input_dim)
for i in range(0, input_dim):
    weights_input[i * q:(i + 1) * q, i] = 2 * np.random.rand(q) - 1 #что за ":"?

wi = weights_input
weights_output = np.zeros((input_dim, reservoir_size))
#5
#Вставьте последовательность в скрытое состояние сети:
def initialize_hidden(reservoir_size, n_steps_prerun, sequence):
    hidden = np.zeros((reservoir_size, 1))
    for t in range(n_steps_prerun):
        
        input = sequence[t].reshape(-1, 1)
        
        hidden = np.tanh(weights_hidden @ hidden + weights_input @ input)
        
    return hidden
#6
def augment_hidden(hidden):
    h_aug = hidden.copy()
    h_aug[::2] = pow(h_aug[::2], 2.0)
    return h_aug
#7
hidden = initialize_hidden(reservoir_size, n_steps_prerun, sequence)
hidden_states = []
targets = []
#8
for t in range(n_steps_prerun, len(sequence) - 1):
    input = np.reshape(sequence[t], (-1, 1))
    target = np.reshape(sequence[t + 1], (-1, 1))
   
    hidden = np.tanh(weights_hidden @ hidden + weights_input @ input)
    hidden = augment_hidden(hidden)
    hidden_states.append(hidden)
    targets.append(target)
#9
targets = np.squeeze(np.array(targets))
hidden_states = np.squeeze(np.array(hidden_states))
print(hidden_states)
#10
#Регрессия для получения линейных весов выходного слоя:
weights_output = (np.linalg.inv(hidden_states.T@hidden_states + regularization * np.eye(reservoir_size)) @ hidden_states.T@targets).T
#11
def predict(sequence, n_steps_predict):
    hidden = initialize_hidden(reservoir_size, n_steps_prerun, sequence) #тут
    input = sequence[n_steps_prerun].reshape((-1, 1))
  
    outputs = []

    for t in range(n_steps_prerun, n_steps_prerun + n_steps_predict):
        hidden = np.tanh(weights_hidden @ hidden + weights_input @ input)
        hidden = augment_hidden(hidden) #тут
        output = weights_output @ hidden
        input = output
        outputs.append(output)
    return np.array(outputs)

x_sim = predict(sequence, x_sim_length) #<-- здесь выбираем, от 0 до каких пор предсказываем

#графики
plt.figure(figsize=(6, 4))
plt.plot(x_train[:x_sim_length, 0], x_train[:x_sim_length, 2], label="Ground truth")
plt.plot(x_sim[:, 0], x_sim[:, 2],'--', label="Reservoir computing estimate")
plt.plot(x_train[0, 0], x_train[0, 2], "ko", label="Initial condition", markersize=8)

plt.legend()
plt.show()

def plot_dimension(dim, name):
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    ax.plot(time_steps, x_train[:, dim])
    ax.plot(time_steps, x_sim[:, dim], "--")
    plt.xlabel("time")
    plt.ylabel(name)
    plt.draw()
    plt.show()

plot_dimension(0, 'x')
plot_dimension(1, 'y')
plot_dimension(2, 'z')

#поиск среднего и максимального отклонений
#поиск среднего и максимального отклонений
x_sim_reservoir_X = np.arange(0.0, x_train_length, dt)
for k in range(x_sim_length):
    x_sim_reservoir_X[k] = x_sim[:, 0][k][0]
    
x_sim_reservoir_Y = np.arange(0.0, x_train_length, dt)
for s in range(x_sim_length):
    x_sim_reservoir_Y[s] = x_sim[:, 1][s][0]

x_sim_reservoir_Z = np.arange(0.0, x_train_length, dt)
for c in range(x_sim_length):
    x_sim_reservoir_Z[c] = x_sim[:, 2][c][0] 
    
variances = []

variances = x_train[:, 0] - x_sim_reservoir_X
variances = np.abs(variances)
print("Макс. откл по ",0, " = ", max(variances))
sum_variance = sum(variances)
x_avg_sum_variance = sum_variance/x_sim_length
print("Ср. откл по ",0, " = ", x_avg_sum_variance)

variances = x_train[:, 1] - x_sim_reservoir_Y
variances = np.abs(variances)
print("Макс. откл по ",1, " = ", max(variances))
sum_variance = sum(variances)
y_avg_sum_variance = sum_variance/x_sim_length
print("Ср. откл по ",1, " = ", y_avg_sum_variance)

variances = x_train[:, 2] - x_sim_reservoir_Z
variances = np.abs(variances)
print("Макс. откл по ",2, " = ", max(variances))
sum_variance = sum(variances)
z_avg_sum_variance = sum_variance/x_sim_length
print("Ср. откл по ",2, " = ", z_avg_sum_variance)








