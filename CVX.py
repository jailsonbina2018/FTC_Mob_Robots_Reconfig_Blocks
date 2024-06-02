
import control as ct
import cvxpy as cp
import numpy as np
from numpy.linalg import inv
from numpy import linalg
import matplotlib.pyplot as plt

# Create data
n = 4 # Actual order of the system
m = 2; p = 2 # Number of inputs and outputs

# System matrices
A = np.array([[1.178, 0.001, 0.511, -0.43],
              [-0.051, 0.661, -0.011, 0.061],
              [0.076, 0.335, 0.560, 0.382],
              [0, 0.335, 0.089, 0.849]])

B = np.array([[0.004, -0.087],
              [0.467, 0.001],
              [0.213, -0.235],
              [0.213, -0.016]])

C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

D = np.zeros((2, 2))

T_s = 0.1 # Sampling time.

T = 15 # Number of samples.

# Create the state space system
sys = ct.ss(A, B, C, D, dt=T_s)

# Initial I/O paths for simulation to obtain some data
u = np.random.rand(2, T)
x = np.zeros((n, T))
x0 = np.random.rand(n, 1)
u0 = np.random.rand(m, 1)

for i in range(n):
    x[i, 0] = sys.A[i, :] @ x0[:, 0] + sys.B[i, :] @ u0[:, 0]

for i in range(T-1): 
    x[:, i+1] = sys.A @ x[:, i] + sys.B @ u[:, i]

U0 = np.hstack((u0, u))
X0 = np.hstack((x0, x))

B_A = np.hstack((B, A))
U_X = np.vstack((U0, X0))

X1 = B_A @ U_X

U0 = cp.Parameter((m, T), value=U0)
X0 = cp.Parameter((n, T), value=X0)
X1 = cp.Parameter((n, T), value=X1)

Q = cp.Variable((T, n), symmetric=False)

constraints = []
constraints.append(Q >> 0)

z = cp.bmat([
    [X0 @ Q, X1 @ Q],
    [Q.T @ X1.T, X0 @ Q]])

constraints.append(z >> 0)

obj = cp.Minimize(cp.norm(X1 - B_A @ U_X))

prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.MOSEK, verbose=False)

print(Q.value)