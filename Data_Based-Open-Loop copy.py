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
t = np.linspace(0, 14, 100)
print(t)

# Sinal de entrada (entrada em etapas)

u = np.ones((2, len(t)))
print(np.shape(u))

# Simular o sistema
t, y = ct.input_output_response(sys, T=t, U=u, X0=np.zeros(4))

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Obtenção da saída x1
x1 = y[0]


# Gráfico 1 - x1
axs[0].plot(t, x1, linestyle='-', color='#120a8f', label='$x_1$',)
axs[0].set_xlabel('Tempo(s)')
axs[0].set_ylabel('$x_1$')
# axs[0].set_title('Posição $x_1$')
axs[0].legend()
axs[0].grid(True)

# Obtenção da saída x1
x2 = y[1]

# Gráfico 1 - x2
axs[1].plot(t, x2, linestyle='-', color='#120a8f', label='$x_2$',)
axs[1].set_xlabel('Tempo(s)')
axs[1].set_ylabel('$x_2$')
# axs[1].set_title('Posição $x_2$')
axs[1].legend()
axs[1].grid(True)
fig.suptitle('Sistema Dinâmico Massa-mola', fontsize=16)
plt.tight_layout()
plt.show()