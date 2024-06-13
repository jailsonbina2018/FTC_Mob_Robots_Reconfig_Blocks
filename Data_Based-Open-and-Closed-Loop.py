import control as ct
import cvxpy as cp
import numpy as np
from numpy.linalg import inv, eig
from numpy import linalg
import matplotlib.pyplot as plt

# System matrices
A = np.array([[1.178, 0.001, 0.511, -0.43],
              [-0.051, 0.661, -0.011, 0.061],
              [0.076, 0.335, 0.560, 0.382],
              [0, 0.335, 0.089, 0.849]])

B = np.array([[0.004, -0.087],
              [0.467, 0.001],
              [0.213, -0.235],
              [0.213, -0.016]])

C = np.array([[1, 0, 0, 0], 
               [0 , 1, 0, 0]])

D = np.zeros((2, 2))

T_s = 0.1  # Sampling time
T = 15     # Number of samples for each state
  

# Estado inicial aleatória
x0 = np.random.rand(4, 1)
u0 = np.random.rand(2, 1)

# Entrada aleatória
u = np.random.rand(2, T) 
x = np.zeros((4, T))



for i in range(4):
     x[i, 0] = np.dot(A[i, :], x0).item() + np.dot(B[i, :], u0).item()
    
# Representação em malha aberta baseada em dados
for i in range(T-1):
     x[:, i+1] = np.dot(A, x[:, i]) + np.dot(B, u[:, i])
     

U0 = np.hstack((u0, u[:, 1:T]))
X0 = np.hstack((x0, x[:, 1:T]))

B_A = np.concatenate((B, A), axis=1)
U_X = np.concatenate((U0, X0), axis=0)
X1 = np.dot(B_A, U_X)

# Configurações do cvxpy

U0 = cp.Parameter((2, T), value=U0)
X0 = cp.Parameter((4, T), value=X0)
X1 = cp.Parameter((4, T), value=X1)

Q = cp.Variable(shape=(15,4), symmetric=False)

constraints = []

z = cp.bmat([
    [X0 @ Q, X1 @ Q], 
    [Q.T @ X1.T, X0 @ Q]])

constraints += [z >> 0]

obj = cp.Minimize(0)

prob = cp.Problem(obj, constraints)

prob.solve(solver=cp.MOSEK, verbose=False)
# print(Q.value)

# Control law
Kn = U0.value @ Q.value @ linalg.inv(X0.value @ Q.value)
Kn = np.array(Kn)
Kn = np.round(Kn, decimals=4)

# Representação em malha fechada baseada em dados

x0_cl = np.random.rand(4, 1)
u0_cl = np.random.rand(2, 1)

# Entrada aleatória
u_cl = np.random.rand(2, T) 
x_cl = np.zeros((4, T))

for i in range(4):
     x_cl[i, 0] = np.dot(A[i, :], x0_cl).item() + np.dot(B[i, :], u0_cl).item()

for i in range(T-1):
     x_cl[:, i+1] = np.dot(A + np.dot(B, Kn), x_cl[:, i])
     

# Gráfico dos resultados

time = np.dot(10, np.arange(0, T*T_s, T_s))
# time = np.arange(0, T*T_s, T_s)

plt.figure(figsize=(10, 6))
# plt.plot(1,1)
plt.subplot(2, 1, 1)
plt.plot(time, x[0, :], label='$x_1$')
plt.plot(time, x[1, :], label='$x_2$')
plt.plot(time, x[2, :], label='$x_3$')
plt.plot(time, x[3, :], label='$x_4$')
plt.xlabel('Tempo (s)')
plt.ylabel('Estados')
plt.legend()
plt.title('Representação em Malha Aberta')

plt.subplot(2, 1, 2)
plt.plot(time, x_cl[0, :], label='$x_1c$')
plt.plot(time, x_cl[1, :], label='$x_2c$')
plt.plot(time, x_cl[2, :], label='$x_3c$')
plt.plot(time, x_cl[3, :], label='$x_4c$')
plt.xlabel('Tempo (s)')
plt.ylabel('Estados')
plt.legend()

plt.title('Representação em Malha Fechada')
plt.tight_layout()
plt.savefig('Data_Based-Open-and-Closed-Loop.png')
plt.savefig('Data_Based-Open-and-Closed-Loop.pdf')
plt.show()
