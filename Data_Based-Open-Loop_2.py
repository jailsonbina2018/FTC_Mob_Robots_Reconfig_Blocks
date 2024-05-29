
import numpy as np
import control as ct
import matplotlib.pyplot as plt

A = np.array([[1.178, 0.001, 0.511, -0.43],
              [-0.051, 0.661, -0.011, 0.061],
              [0.076, 0.335, 0.560, 0.382],
              [0, 0.335, 0.089, 0.849]])

B = np.array([[0.004, -0.087],
              [0.467, 0.001],
              [0.213, -0.235],
              [0.213, -0.016]])

C = np.eye(4)  # Identity matrix for output
D = np.zeros((4, 2))  # Zero matrix for direct feedthrough

system = ct.ss(A, B, C, D, inputs=['d'], states=['x1', 'x2', 'x3', 'x4'], outputs=['y1', 'y2'])
timepts = np.linspace(0, 60, 100)
t, y = ct.input_output_response(sys=system, T=timepts, U=1, X0=[0, 0, 0, 0])
fig, axs = plt.subplots(1, 1, figsize=(12, 4))
x1 = y[0]
axs.plot(t, x1, linestyle='-', color='#120a8f', label='$x1$')
axs.set_xlabel('tempo(s)')
axs.set_ylabel('$x1$')
axs.set_title('Sistema em malha aberta')
axs.legend()
axs.grid(True)
plt.show()
