import control as ct
import numpy as np
import matplotlib.pyplot as plt

# Criar dados
i = 1 # Indica o momento em que a primeira amostra do sinal é obtida.
t = 1 # Número de amostras em cada coluna.
N = 1 # Número de amostras de sinal por cada linha.
T = 15 # 
n_input = 2; n_output = 2 # Número de entradas e saídas


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
sys = ct.ss(A, B, C, D, T_s)

# Trajetórias iniciais de E/S para simulação para obter alguns dados
ui = np.random.rand(n_input, T) # entrada inicial
print(np.shape(ui))
xi0 = np.random.rand(n_input, 1)
xi = np.zeros((n_input, n_input))
xi[:, 0] = xi0.squeeze()

for i in range(n_input-1):
    xi[:, i+1] = sys.A @ xi[:, i] + sys.B @ ui[:, i]

yi = sys.C @ xi + sys.D @ ui
print(yi)
