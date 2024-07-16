import control as ct
import cvxpy as cp
import numpy as np
from numpy.linalg import inv, eig
from numpy import linalg
import matplotlib.pyplot as plt

# Matrizes dos sistema
Atr = np.array([[-0.5, 1.4, 0.4],
                [-0.9, 0.3, -1.5],
                [1.1, 1, -0.4]])

Btr = np.array([[0.1, -0.3],
                [-0.1, -0.7],
                [0.7, -1]])

Bw = np.identity(3)

Ctr = np.identity(3)

Dtr = 0
Dw = 0

Ln = 3  # Número de linhas.
Nc = 20  # Número de amostras para cada estado T >= (m + 1)n + m
T_s = 0.1  # Tempo de amostragem

n = 3
m = 2
t = 1

# Estado inicial aleatória
x0 = np.random.rand(Ln, 1)
u0 = np.random.rand(m, 1) - 1  # Entrada u(k) uniformemente de [-1, 1]
w_medin = 0.02
w0 = np.dot(w_medin, np.random.rand(n, 1))

# Entrada aleatória
u = np.random.rand(m, Nc) - 1  # Entrada u(k) uniformemente de [-1, 1]
x = np.zeros((3, Nc))
w = np.dot(w_medin, np.random.rand(n, Nc))

for j in range(n):
    x[j, 0] = np.dot(Atr[j, :], x0).item() + np.dot(Bw[j, :], w0).item() + np.dot(Btr[j, :], u0).item()

# Representação em malha aberta baseada em dados
for k in range(Nc-1):
    x[:, k+1] = np.dot(Atr, x[:, k]) + np.dot(Bw, w[:, k]) + np.dot(Btr, u[:, k])

Xs0 = np.hstack((x0, x[:, 1:Nc]))  # X
Xds1 = np.hstack((x[:, 1:2], x[:, 1:Nc]))  # X+

# Configurações do cvxpy
gm = 2.4

X0 = cp.Parameter((n, Nc), value=Xs0)
X1 = cp.Parameter((n, Nc), value=Xds1)

Y = cp.Variable(shape=(n, n), symmetric=True)
M = cp.Variable(shape=(Nc, n), symmetric=False)

Q = - np.dot(np.square(gm), np.identity(n))
S = 0
R = np.identity(n)

QW = - np.identity(n)
SW = cp.Parameter((n, Nc), value=np.zeros((n, Nc)))
RW = cp.Parameter((Nc, Nc), value=np.linalg.inv(np.dot(np.square(w_medin), np.identity(Nc))))
BW = cp.Parameter((n, n), value=Bw)

constraints = []

H11 = -Y
H12 = -M.T @ SW.T
H13 = M.T @ X1.T
H14 = M.T

H21 = -SW @ M
H22 = QW
H23 = BW.T
H24 = 0

H31 = X1 @ M
H32 = BW
H33 = -Y
H34 = 0

H41 = M
H42 = 0
H43 = 0
H44 = - RW


H = cp.bmat([
    [H11, H12, H13, H14],
    [H21, H22, H23, H24],
    [H31, H32, H33, H34],
    [H41, H42, H43, H44]])

obj = cp.Minimize(cp.sum([cp.sum(cp.square(M))]))
constraints += [H << 0]

prob = cp.Problem(obj, constraints)

prob.solve(solver=cp.MOSEK, verbose=False)

print(Y.value)
print(M.value)