import numpy as np
import cvxpy as cp

# Definição da matriz A
A = np.array([[0, 1, 0],
              [-30, -5, 2.5],
              [0, -20.83, -16.67]])

# Definição da matriz N como uma matriz identidade
N = np.eye(3)

# Definição da variável de decisão M (matriz simétrica)
M = cp.Variable((3, 3), symmetric=True)

# Definição da restrição da equação de Lyapunov
lyapunov_eq = A.T @ M + M @ A + N

# Definição do problema de otimização
constraints = [lyapunov_eq == -N, M >> 0]
problem = cp.Problem(cp.Minimize(0), constraints)

# Resolver o problema
problem.solve()

# Verificar se M é positiva definida
positive_definite = np.all(np.linalg.eigvals(M.value) > 0)

M_value = M.value
print("Matriz M:")
print(M_value)
print("M é positiva definida:", positive_definite)
