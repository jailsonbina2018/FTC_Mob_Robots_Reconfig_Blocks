import control as ct
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


A = np.array([[1.178, 0.001, 0.511, -0.43],
     [-0.051, 0.661, -0.011, 0.061],
     [0.076, 0.335, 0.560, 0.382],
     [0, 0.335, 0.089, 0.849]])

B = np.array([[0.004, -0.087],
     [0.467, 0.001],
     [0.213, -0.235],
     [0.213, -0.016]])

# C =  np.eye(4)
C = np.array([[1, 0, 1, -1], [0, 1, 0, 0]])

# D = np.zeros((4, 2))
D = 0

# Definir o sistema de espaço de estado
sys = ct.ss(A, B, C, D)

# print(sys)

# Vetor de tempo para simulação
# t = np.arange(0, 10, dt)
t = np.linspace(0,15,15)

# Sinal de entrada (entrada em etapas)

u = np.ones((2, len(t)))

# Simular o sistema[1, 1]
t, y = ct.input_output_response(sys, T=t, U=u, X0=np.zeros(4))

print(y)

# C_inv = inv(C)

""" x = inv(C).dot(y)
print(np.shape(x))
print(x) """