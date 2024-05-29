import control as ct
import numpy as np
import matplotlib.pyplot as plt

A = [[1.178, 0.001, 0.511, -0.43],
     [-0.051, 0.661, -0.011, 0.061],
     [0.076, 0.335, 0.560, 0.382],
     [0, 0.335, 0.089, 0.849]]

B = [[0.004, -0.087],
     [0.467, 0.001],
     [0.213, -0.235],
     [0.213, -0.016]]

C =  np.eye(4)

D = np.zeros((4, 2))

# Definir o sistema de espaço de estado
sys = ct.ss(A, B, C, D)

# print(sys)

# Vetor de tempo para simulação
t = np.linspace(0, 14, 1000)
print(t)

# Sinal de entrada (entrada em etapas)

u = np.ones((2, len(t)))

# Simular o sistema
t, y = ct.input_output_response(sys, T=t, U=u, X0=np.zeros(4))

# Gráfico da resposta
x1 = y[0]

plt.figure()
plt.plot(t, x1.T, linestyle='-', color='#120a8f', label='$x_1$',)
plt.xlabel('Tempo(s)')
plt.ylabel('$x_1$')
plt.title('Estado $x_1$')
plt.legend()
plt.grid(True)
plt.show()

