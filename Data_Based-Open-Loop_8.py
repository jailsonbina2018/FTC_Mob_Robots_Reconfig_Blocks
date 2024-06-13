import control as ct
import cvxpy as cp
import numpy as np
from numpy.linalg import inv, eig
from numpy import linalg
import matplotlib.pyplot as plt

# Criar dados
n = 4 # Ordem real do sistema
m = 2; p = 2 # Número de entradas e saídas


# Matrizes do sistema

A = np.array([[1.178, 0.001, 0.511, -0.43],
              [-0.051, 0.661, -0.011, 0.061],
              [0.076, 0.335, 0.560, 0.382],
              [0, 0.335, 0.089, 0.849]])

B = np.array([[0.004, -0.087],
              [0.467, 0.001],
              [0.213, -0.235],
              [0.213, -0.016]])

C =  np.array([[1, 0, 0, 0], 
               [0 , 1, 0, 0]])

D = np.zeros((2, 2))

K = np.array([[0.7610, -1.1363, 1.6945, -1.8123], 
              [3.5351, 0.4827, 3.3014, -2.6215]])


T_s = 0.1 # Tempo de amostragem.

T = 15 # Quantidade de amostras.


# Criar o sistema de espaço de estado
sys = ct.ss(A, B, C, D, dt=T_s)

# Trajetórias iniciais de E/S para simulação para obter alguns dados

u = np.random.rand(2, 15) - 1
x = np.zeros((4, 15))
x0 = np.random.rand(4, 1)
u0 = np.random.rand(2, 1)

for i in range(n):
     x[i, 0] = np.dot(sys.A[i, :], x0).item() + np.dot(sys.B[i, :], u0).item()
     

for i in range(T-1):
     x[:, i+1] = np.dot(sys.A, x[:, i]) + np.dot(sys.B, u[:, i])
    

plt.plot(1,1)
plt.plot(np.arange(T), x[0, :], label='$x_1$')
plt.plot(np.arange(T), x[1, :], label='$x_2$')
plt.plot(np.arange(T), x[2, :], label='$x_3$')
plt.plot(np.arange(T), x[3, :], label='$x_4$')
plt.xlabel('Tempo (s)')
plt.ylabel('Valor do Estado')
plt.title('Trajetória do estado')
plt.legend()
plt.tight_layout() 
plt.show()