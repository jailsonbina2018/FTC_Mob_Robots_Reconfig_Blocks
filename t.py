import numpy as np
import control

# Define the state-space matrices
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

## Create the state-space model
ss_1 = control.ss(A, B, C, D)

## Simulate the system
t = np.linspace(0, 10, 1000)  # Time vector
u = np.zeros((2, 1000))  # Input vector (zeros)
t, y= control.input_output_response(ss_1, t, u)

## Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, y[:, 0], label='State 1')
plt.plot(t, y[:, 1], label='State 2')
plt.plot(t, y[:, 2], label='State 3')
plt.plot(t, y[:, 3], label='State 4')
plt.xlabel('Time (s)')
plt.ylabel('State values')
plt.legend()
plt.title('State Response')

plt.subplot(2, 1, 2)
plt.plot(t, u[0, :], label='Input 1')
plt.plot(t, u[1, :], label='Input 2')
plt.xlabel('Time (s)')
plt.ylabel('Input values')
plt.legend()
plt.title('Input')

plt.tight_layout()
plt.show()