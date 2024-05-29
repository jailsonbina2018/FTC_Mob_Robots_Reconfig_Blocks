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

C =  np.eye(4)

D = np.zeros((4, 2))

dt=0.1
# Definir o sistema de espaço de estado
sys = ct.ss(A, B, C, D, dt=dt)

# print(sys)

# Vetor de tempo para simulação
t = np.arange(0, 10, dt)
# t = np.linspace(0, 15, 15)

# Sinal de entrada (entrada em etapas)

u = np.ones((2, len(t)))

# Simular o sistema
t, Y_0T= ct.input_output_response(sys, T=t, U=u, X0=np.zeros(4))

C_inv = inv(C)
# print(np.shape(C_inv))
# print(np.shape(Y_0T))
X_0T = C_inv.dot(Y_0T)

U_01T = u
# print(x)
# print(np.shape(X_0T))
# print(np.shape(u))
# u_T = u.T
# print(np.shape(u_T))
B_A = np.concatenate((B, A), axis=1)
# print(np.shape(B_A))
U_X = np.concatenate((U_01T, X_0T), axis=0)
# print(np.shape(U_X))
X_1T = B_A.dot(U_X)
# print(np.shape(X_1T))
# print(A)
X_0T = cp.Parameter((4, 15), value=X_0T)
X_1T = cp.Parameter((4, 15), value=X_1T)

Q = cp.Variable(shape=(2,2), symmetric=True)

constraints = []

constraints = [Q >> 0]

z = cp.bmat([
    [X_0T @ Q, X_1T @ Q], 
    [Q.T @ X_1T.T+ B @ Y, Q]])

constraints += [z >> 0]

obj = cp.Minimize(0)

prob = cp.Problem(obj, constraints)

prob.solve(solver=cp.MOSEK, verbose=False)