import control as ct
import numpy as np
import matplotlib.pyplot as plt

# Definição dos Parâmetros
m1 = 1.
m2 = 0.5
k1 = 1.0
k2 = 1.0
c0 = 2.0

A = np.array([[0., 0., 1., 0.], 
     [0., 0., 0., 1.], 
     [-((k1 + k2) / m1), (k2 / m1), - (c0 / m1), 0.], 
     [(k2 / m2), - (k2 / m2), 0., - (c0 / m2)]])

B = np.array([[0.], [0.], [1. / m1], [0.]])

C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

D = np.array(0)

""" # Define the state-space matrices
A = np.array([[1.178, 0.001, 0.511, -0.43],
              [-0.051, 0.661, -0.011, 0.061],
              [0.076, 0.335, 0.560, 0.382],
              [0, 0.335, 0.089, 0.849]])

B = np.array([[0.004, -0.087],
              [0.467, 0.001],
              [0.213, -0.235],
              [0.213, -0.016]])

C = np.eye(4)  # Identity matrix for output
D = np.zeros((4, 2))  # Zero matrix for direct feedthrough """



# Definição do sistema massa-mola
system = ct.ss(A, B, C, D, name='Sistema Massa-mola', inputs=('d'), states=('x1', 'x2', 'x1_dot', 'x2_dot'), outputs=('x1', 'x2',))

# Simulação do sistema

# Tempo de simulação: 0 a 60 segundos, com 100 pontos

timepts = np.linspace(0, 60, 100)
u = np.ones((2, len(timepts)))

t, y = ct.input_output_response(sys=system, T=timepts, U=[1], X0=[0, 0, 0, 0],)

# Apresentação dos resultados

# Criação e apresentação dos gráficos
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Obtenção da saída x1
x1 = y[system.find_output('x1')]

# Gráfico 1 - x1
axs[0].plot(t, x1, linestyle='-', color='#120a8f', label='$\delta i_L$',)
axs[0].set_xlabel('Tempo(s)')
axs[0].set_ylabel('$x_1$')
axs[0].set_title('Posição $x_1$')
axs[0].legend()
axs[0].grid(True)

# Obtenção da saída x1
x2 = y[system.find_output('x2')]

# Gráfico 1 - x2
axs[1].plot(t, x2, linestyle='-', color='#120a8f', label='$\delta v_C$',)
axs[1].set_xlabel('Tempo(s)')
axs[1].set_ylabel('$x_2$')
axs[1].set_title('Posição $x_2$')
axs[1].legend()
axs[1].grid(True)
fig.suptitle('Sistema Dinâmico Massa-mola', fontsize=16)
plt.tight_layout()
plt.show()