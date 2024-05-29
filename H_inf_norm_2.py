import cvxpy as cp
import numpy as np
import math

# Definição dos Parâmetros
m1 = 1.0
m2 = 0.5
k1 = 1.0
k2 = 1.0
c0 = 2.0

A = cp.Parameter((4, 4), value=[[0., 0., - ((k1 + k2) / m1), (k2 /
m2)],[0., 0., (k2 / m1), - (k2 / m2)], [1., 0., - (c0 / m1), 0],[0., 1., 0., -(c0 / m2)]])
B = cp.Parameter((4, 1), value=[[0., 0., 1. / m1, 0.]])
C = cp.Parameter((1, 4), value=[[0], [1], [0], [0]])
D = cp.Parameter((1, 1), value=[[0]])
I = cp.Parameter((1, 1), value=np.identity(1))   

print('Parâmetros:\n')
print('A = \n', A.value)
print('B', B.value)
print('C', C.value)
print('D', D.value)

# Definição das Variáveis
P = cp.Variable((4, 4), symmetric=True)
mu = cp.Variable()
M = cp.bmat([[A.T @ P + P @ A + C.T @ C, P @ B + C.T @ D],
[B.T @ P + D.T @ C, D.T @ D - mu * I]])

# Definição do Problema: Objetivo e Restrições
obj = cp.Minimize(mu)
constraints = [P >> 0, M << 0]

prob = cp.Problem(obj, constraints)

# Resolução do Problema usando o solver MOSEK
prob.solve(solver=cp.MOSEK, verbose=False)
print("\nValor ótimo:", prob.value, '\n')

# Apresentação dos Resultados
print('P = \n', P.value)
print("mu = ", mu.value, '\n')
print("Norma Hinf = ", math.sqrt(mu.value), '\n')

# Optimal value of P
P_opt = P.value

# Compute the optimal state feedback gain K
K_opt = -np.linalg.inv(D.T @ D - mu.value * I.value) @ (B.T @ P_opt + D.T @ C.value)
