import control as ct
import cvxpy as cp
import numpy as np
from numpy.linalg import inv
from numpy import linalg
import matplotlib.pyplot as plt

# Criar dados
n = 4 # Ordem real do sistema
m = 2; p = 2 # Número de entradas e saídas

t = 1
T = 15
L = 0
#m = 0
#n = 0



# Matrizes do sistema

A = np.array([[1.178, 0.001, 0.511, -0.43],
     [-0.051, 0.661, -0.011, 0.061],
     [0.076, 0.335, 0.560, 0.382],
     [0, 0.335, 0.089, 0.849]])

B = np.array([[0.004, -0.087],
     [0.467, 0.001],
     [0.213, -0.235],
     [0.213, -0.016]])

C =  np.array([[1, 0, 0, 0], [0 , 1, 0, 0]])

D = np.zeros((2, 2))

T_s = 0.1

# Criar o sistema de espaço de estado
sys = ct.ss(A, B, C, D, dt=T_s)

# Trajetórias iniciais de E/S para simulação para obter alguns dados

ui = np.random.rand(2, T) # entrada inicial
# print(f'xi0 = {ui}')

xi0 = np.random.rand(4, T)
# print(f'xi0 = {xi0}')
#
xi = np.zeros((4, T))
# print(f'xi = {xi}')

xi[:, ] = xi0.squeeze()
# print(f'xi[:, ] = {xi[:, ]}')

for i in range(T-1):
    xi[:, i+1] = sys.A @ xi[:, i] + sys.B @ ui[:, i]

print(f'xi[:, i+1] = {xi[:, i+1]}')
print(np.shape(xi[:, i+1]))




