import control as ct
import cvxpy as cp
import numpy as np
from numpy.linalg import inv
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

C =  np.array([[1, 0, 0, 0], [0 , 1, 0, 0]])

D = np.zeros((2, 2))

T_s = 0.1

# Criar o sistema de espaço de estado
sys = ct.ss(A, B, C, D, dt=T_s)

# Trajetórias iniciais de E/S para simulação para obter alguns dados
ui = np.random.rand(m, n) # entrada inicial

xi0 = np.random.rand(n, 1)

xi = np.zeros((n, n))

xi[:, 0] = xi0.squeeze()
print(np.shape(xi[0,:]))

for i in range(n-1):
    xi[:, i+1] = sys.A @ xi[:, i] + sys.B @ ui[:, i]
          
X_0T = xi
U_01T = ui


B_A = np.concatenate((B, A), axis=1)

U_X = np.concatenate((U_01T, X_0T), axis=0)

X_1T = B_A.dot(U_X)

X_0T = cp.Parameter((4, 4), value=X_0T)
X_1T = cp.Parameter((4, 4), value=X_1T)
U_01T = cp.Parameter((2, 4), value=U_01T)

Q = cp.Variable(shape=(4,4), symmetric=True)

constraints = []

constraints = [Q >> 0]

z = cp.bmat([
    [X_0T @ Q, X_1T @ Q], 
    [Q.T @ X_1T.T, X_0T @ Q]])

constraints += [z >> 0]

obj = cp.Minimize(0)

prob = cp.Problem(obj, constraints)

prob.solve(solver=cp.MOSEK, verbose=False)


print(f"K = {U_01T.value @ Q.value @ linalg.inv(X_0T.T.value @ Q.value)}")