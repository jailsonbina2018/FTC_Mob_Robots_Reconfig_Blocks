
import numpy as np
import matplotlib.pyplot as plt

# Define the matrices A and B
A = np.array([[1.178, 0.001, 0.511, -0.43],
              [-0.051, 0.661, -0.011, 0.061],
              [0.076, 0.335, 0.560, 0.382],
              [0, 0.335, 0.089, 0.849]])

B = np.array([[0.004, -0.087],
              [0.467, 0.001],
              [0.213, -0.235],
              [0.213, -0.016]])

# Generate random input u with amplitude from 0 to 1 and 15 values
np.random.seed(0)  # For reproducibility
u = np.random.rand(15, 2)

# Generate random initial state x0
x0 = np.random.rand(4)

# Sample time Ts = 0.1s
Ts = 0.1

# Number of time steps
n_steps = 15

# Initialize the state and output arrays
x = np.zeros((n_steps, 4))
y = np.zeros((n_steps, 2))

# Set the initial state
x[0] = x0

# Simulate the system
for i in range(1, n_steps):
    x[i] = np.dot(A, x[i-1]) + np.dot(B, u[i-1])

# Plot the state and output
plt.figure(figsize=(12, 6))

plt.plot(1, 1)
plt.plot(x)
plt.title("State")
plt.xlabel("Time Steps")
plt.ylabel("State Values")
plt.legend(["x1", "x2", "x3", "x4"])
#plt.tight_layout()
plt.show()