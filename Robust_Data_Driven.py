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

C = np.identity(3)

D = 0
Dw = 0

L = 3  # Número de linhas.
N = 20  # Número de amostras para cada estado T >= (m + 1)n + m
T_s = 0.1  # Tempo de amostragem

n = 3
m = 2
t = 1

# Estado inicial aleatória
x0 = np.random.rand(L, 1)
u0 = np.random.rand(m, 1) - 1  # Entrada u(k) uniformemente de [-1, 1]
w_medin = 0.02
w0 = np.dot(w_medin, np.random.rand(n, 1))

# Entrada aleatória
u = np.random.rand(m, N) - 1  # Entrada u(k) uniformemente de [-1, 1]
x = np.zeros((3, N))
w = np.dot(w_medin, np.random.rand(n, N))

for j in range(n):
    x[j, 0] = np.dot(Atr[j, :], x0).item() + np.dot(Bw[j, :], w0).item() + np.dot(Btr[j, :], u0).item()

# Representação em malha aberta baseada em dados
for k in range(N-1):
    x[:, k+1] = np.dot(Atr, x[:, k]) + np.dot(Bw, w[:, k]) + np.dot(Btr, u[:, k])


X0 = np.hstack((x0, x[:, 1:N-1]))  # X
X1 = np.hstack((x[:, 1:N]))  # X+

print(f'X0 = {np.shape(X0)}')
print(f'X1 = {np.shape(X1)}')

"""
# Configurações do cvxpy
gm = 2.4

p = cp.Variable()

X0 = cp.Parameter((n, N), value=X0)
X1 = cp.Parameter((n, N), value=X1)

Y = cp.Variable(shape=(m, m), symmetric=True)
M = cp.Variable(shape=(N, n), symmetric=False)

Q = - np.dot(np.square(gm), np.identity(n))
S = 0
R = np.identity(n)

Qw = - np.identity(n)
Sw = np.zeros((n, n))
Rw = np.dot(np.square(w_medin), np.identity(N))

constraints = []

H11 = -Y
H12 = -M.T @ Sw.T
H13 = M.T @ X1.T
H14 = M.T

H21 = -Sw @ M
H22 = Qw
H23 = Bw.T
H24 = 0

H31 = X1 @ M
H32 = Bw
H33 = -Y
H34 = 0

H41 = M
H42 = 0
H43 = 0
H44 = - np.linalg.inv(Rw)


H = cp.bmat([
    [H11, H12, H13],
    [H21, H22, H23],
    [H31, H32, H33]])


obj = cp.Minimize(p)
constraints += [Y >> 0]
constraints += [M >> 0]

prob = cp.Problem(obj, constraints)

prob.solve(solver=cp.MOSEK, verbose=False)

print(Y.value)
print(M.value)

"""

# Lei de controle

""" 
Kn = U0.value @ Q.value @ linalg.inv(X0.value @ Q.value)
Kn = np.array(Kn)
Kn = np.round(Kn, decimals=4) 

"""


"""
# Representação em malha fechada baseada em dados

x0_cl = np.random.rand(n, 1)
u0_cl = np.random.rand(m, 1)

# Entrada aleatória
u_cl = np.random.rand(m, T)
x_cl = np.zeros((n, T))

for i in range(4):
     x_cl[i, 0] = np.dot(A[i, :], x0_cl).item() + np.dot(B[i, :], u0_cl).item()

for i in range(T-1):
     # x_cl[:, i+1] = np.dot(A + np.dot(B, Kn), x_cl[:, i])
     x_cl[:, i+1] = np.dot(A + np.dot(B, Kn), x_cl[:, i])
     # z(:, i) = np.dot(A + np.dot(B, Kn), x_cl[:, i])


 """
# Gráfico dos resultados

"""
time = np.dot(10, np.arange(0, N*T_s, T_s))
# time = np.arange(0, T*T_s, T_s)

plt.figure(figsize=(10, 6))
plt.plot(1,1)
# plt.subplot(2, 1, 1)
plt.plot(time, x[0, :], label='$x_1$')
plt.plot(time, x[1, :], label='$x_2$')
plt.plot(time, x[2, :], label='$x_3$')
# plt.plot(time, x[3, :], label='$x_4$')
plt.xlabel('Tempo (s)')
plt.ylabel('Estados')
plt.legend()
plt.title('Representação em Malha Aberta LQR')

"""


"""
plt.subplot(2, 1, 2)
plt.plot(time, x_cl[0, :], label='$x_1c$')
plt.plot(time, x_cl[1, :], label='$x_2c$')
plt.plot(time, x_cl[2, :], label='$x_3c$')
plt.plot(time, x_cl[3, :], label='$x_4c$')
plt.xlabel('Tempo (s)')
plt.ylabel('Estados')
plt.legend()
plt.title('Representação em Malha Fechada LQR')


plt.tight_layout()
# plt.savefig('Data_Based-Open-and-Closed-Loop_LQR.png')
# plt.savefig('Data_Based-Open-and-Closed-Loop_LQR.pdf')
plt.show()

"""
