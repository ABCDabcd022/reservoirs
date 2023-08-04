from reservoirpy.nodes import Reservoir, Ridge
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import reservoirpy as rpy

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
dt = 0.01

def f(state, t):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

x_train_length = 60.0 

state0 = [1.0, 1.0, 1.0]
time_steps = np.arange(0.0, x_train_length, dt) 

X = odeint(f, state0, time_steps) 

rpy.verbosity(0)
rpy.set_seed(42)  

reservoir1 = Reservoir(100, lr=0.5, sr=0.9) 

X_train = X[:3000]
Y_train = X[1:3001]

reservoir1.run(X_train, reset=True)

reservoir1 = Reservoir(100, lr=0.5, sr=0.9)
ridge1 = Ridge(ridge=1e-7)

esn_model = reservoir1 >> ridge1
esn_model = esn_model.fit(X_train, Y_train, warmup=10)
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