import control as ct
import cvxpy as cp
import numpy as np
from numpy.linalg import inv, eig
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

C =  np.array([[1, 0, 0, 0], 
               [0 , 1, 0, 0]])

D = np.zeros((2, 2))

K = np.array([[0.7610, -1.1363, 1.6945, -1.8123], 
              [3.5351, 0.4827, 3.3014, -2.6215]])


T_s = 0.1 # Tempo de amostragem.

T = 15 # Quantidade de amostras.


# Criar o sistema de espaço de estado
sys = ct.ss(A, B, C, D, dt=T_s)

# Trajetórias iniciais de E/S para simulação para obter alguns dados

u = np.random.rand(2, 15)
x = np.zeros((4, 15))
x0 = np.random.rand(4, 1)
u0 = np.random.rand(2, 1)

for i in range(n):
     x[i, 0] = np.dot(sys.A[i, :], x0).item() + np.dot(sys.B[i, :], u0).item()
     # x[i, 0] = sys.A[i, :] @ x0 + sys.B[i, :] @ u0
     # x[i, 0] = np.dot(sys.A[i, :].flatten(), x0) + np.dot(sys.B[i, :].flatten(), u0)
     # x[i, 0] = np.dot(sys.A[i, :], x0) + np.dot(sys.B[i, :], u0)

for i in range(T-1):
     x[:, i+1] = np.dot(sys.A, x[:, i]) + np.dot(sys.B, u[:, i]) 
     # x[:, i+1] = sys.A @ x[:, i] + sys.B @ u[:, i]
     

U0 = np.hstack((u0, u[:, 1:15]))
X0 = np.hstack((x0, x[:, 1:15]))

B_A = np.concatenate((B, A), axis=1)
U_X = np.concatenate((U0, X0), axis=0)

# X1 = B_A @ U_X
X1 = np.dot(B_A, U_X)

# Configurações do cvxpy

U0 = cp.Parameter((2, 15), value=U0)
X0 = cp.Parameter((4, 15), value=X0)
X1 = cp.Parameter((4, 15), value=X1)

Q = cp.Variable(shape=(15,4), symmetric=False)

constraints = []

# constraints = [Q >> 0]

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


# print(np.shape(A))
# print(np.shape(B))
# print(np.shape(K))

# abs_Kn = np.abs(np.linalg.eig(A + B @ Kn)[0])
# abs_K = np.absolute(eig(A + np.dot(B,K)))

S_K = A + np.dot(B, K)
S_Kn = A + np.dot(B, Kn)

abs_K = np.linalg.norm(S_K)
abs_Kn = np.linalg.norm(S_Kn)

# print(f'abs_K = {abs_K}')
# print(f'abs_Kn = {abs_Kn}')


# abs_Kn = np.absolute(eig(A + B @ Kn))
# print(f'abs_Kn = {abs_Kn}')
# Simulate the closed-loop system



u = np.random.rand(2, 15)
x = np.zeros((4, 15))
x0 = np.random.rand(4, 1)
u0 = np.random.rand(2, 1)

for i in range(n):
     x[i, 0] = np.dot(sys.A[i, :], x0).item() + np.dot(sys.B[i, :], u0).item()
     # x[i, 0] = sys.A[i, :] @ x0 + sys.B[i, :] @ u0
     # x[i, 0] = np.dot(sys.A[i, :].flatten(), x0) + np.dot(sys.B[i, :].flatten(), u0)
     # x[i, 0] = np.dot(sys.A[i, :], x0) + np.dot(sys.B[i, :], u0)
""" 
for i in range(T-1):
     x[:, i+1] = np.dot(sys.A, x[:, i]) + np.dot(sys.B, u[:, i]) 
     # x[:, i+1] = sys.A @ x[:, i] + sys.B @ u[:, i]
     
 """




xn = np.zeros((4, T))
un = np.zeros((2, T))
#un = np.dot(Kn, x)
Kp = 0.5

#AK = A - np.dot(Kp, np.dot(B, Kn))

#BK = np.dot(Kp, B)

r = np.ones((2, T))
#print(r)
#sys_c = ct.ss(AK, BK, C, D, dt=T_s)
#x[:, 0] = x.flatten()
#u[:, 0] = u.flatten()


for i in range(T-1):

    un[:, i+1] = np.dot(Kp, r[:, i]) - np.dot(Kp, np.dot(Kn, x[:, i]))
    xn[:, i+1] = np.dot(sys.A, xn[:, i]) + np.dot(sys.B, un[:, i])


# Plot the results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(np.arange(T), x[0, :], label='$x_1$')
plt.plot(np.arange(T), x[1, :], label='$x_2$')
plt.plot(np.arange(T), x[2, :], label='$x_3$')
plt.plot(np.arange(T), x[3, :], label='$x_4$')
plt.xlabel('Tempo (s)')
plt.ylabel('Valor do Estado')
plt.title('Trajetória do estado')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.arange(T), r[0, :], label='$u_1$')
plt.plot(np.arange(T), r[1, :], label='$u_2$')
plt.xlabel('Tempo (s)')
plt.ylabel('Valor de controle')
plt.title('Trajetória de controle')
plt.legend()

plt.tight_layout() 
plt.show()