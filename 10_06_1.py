import numpy as np
import matplotlib.pyplot as plt

# Given system matrices
A = np.array([[1.178, 0.001, 0.511, -0.43],
              [-0.051, 0.661, -0.011, 0.061],
              [0.076, 0.335, 0.560, 0.382],
              [0, 0.335, 0.089, 0.849]])

B = np.array([[0.004, -0.087],
              [0.467, 0.001],
              [0.213, -0.235],
              [0.213, -0.016]])

# Random initial state
x0 = np.random.rand(4, 1)

# Random input with amplitude from 0 to 1 for 15 time steps
num_steps = 15
u = np.random.rand(num_steps, 2)  # Random values between 0 and 1

# Initialize state array
x = np.zeros((num_steps + 1, 4, 1))
x[0] = x0

# Simulate the system
for k in range(num_steps):
    x[k + 1] = A @ x[k] + B @ u[k].reshape(-1, 1)



# Extract state values for plotting
x1 = x[:, 0, 0]
x2 = x[:, 1, 0]
x3 = x[:, 2, 0]
x4 = x[:, 3, 0]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x1, label='x1')
plt.plot(x2, label='x2')
plt.plot(x3, label='x3')
plt.plot(x4, label='x4')
plt.xlabel('Time step')
plt.ylabel('State value')
plt.title('State Space System Simulation')
plt.legend()
plt.grid(True)
plt.show()
