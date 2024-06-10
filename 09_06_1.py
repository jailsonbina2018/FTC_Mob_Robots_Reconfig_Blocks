import numpy as np
import matplotlib.pyplot as plt

# Define the state-space matrices
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

# Initial state
x0 = np.array([0, 0, 0, 0])

# Sampling time
Ts = 0.1

# Number of time steps
n_steps = 15

# Time vector
t = np.linspace(0, (n_steps-1)*Ts, n_steps)

# Input (zero input for open-loop simulation)
u = np.zeros((2, n_steps))

# Initialize state vector
x = np.zeros((A.shape[0], n_steps))
x[:, 0] = x0

# Simulate the system
for k in range(1, n_steps):
    x[:, k] = A @ x[:, k-1] + B @ u[:, k-1]

# Plot the states
plt.figure(figsize=(10, 6))
plt.plot(t, x[0, :], label='State x1')
plt.plot(t, x[1, :], label='State x2')
plt.xlabel('Time [s]')
plt.ylabel('State value')
plt.title('State Trajectories of the Discrete-Time System')
plt.legend()
plt.grid(True)
plt.show()
