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
C = cp.Parameter((2, 4), value=[[0, 1], [1, 0], [0, 0], [0, 0]])

print('Parâmetros:\n')
print('A = \n', A.value)
print('B', B.value)
print('C', C.value)

# Definição das Variáveis
rho = cp.Variable()
W = cp.Variable((4, 4), symmetric=True)
# Definição do Problema: Objetivo e Restrições
obj = cp.Minimize(rho)
constraints = [W >> 0]
constraints += [rho >= cp.trace(C @ W @ C.T)]
constraints += [A @ W + W @ A.T + B @ B.T << 0]
prob = cp.Problem(obj, constraints)

# Resolução do Problema usando o solver MOSEK
prob.solve(solver=cp.MOSEK, verbose=False)
print("\nValor ótimo:", prob.value, '\n')

39 # Apresentac¸a˜o dos Resultados
print('W = \n', W.value)
print("rho = ", rho.value, '\n')
print("Norma H2 = ", math.sqrt(rho.value), '\n')