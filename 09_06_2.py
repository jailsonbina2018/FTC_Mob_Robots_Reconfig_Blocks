import numpy as np
import matplotlib.pyplot as plt

n = 4
# System matrices
A = np.array([[1.178, 0.001, 0.511, -0.43],
              [-0.051, 0.661, -0.011, 0.061],
              [0.076, 0.335, 0.560, 0.382],
              [0, 0.335, 0.089, 0.849]])

B = np.array([[0.004, -0.087],
              [0.467, 0.001],
              [0.213, -0.235],
              [0.213, -0.016]])

C = np.array([[1, 0, 0, 0], 
               [0 , 1, 0, 0]])

D = np.zeros((2, 2))

T_s = 0.1  # Sampling time
T = 15     # Number of samples for each state
u = np.random.rand(T, 2)  # Random input

# Initial state
# x0 = np.array([0, 0, 0, 0])

u = np.random.rand(2, T)
x = np.zeros((4, T))
x0 = np.random.rand(4, 1)
u0 = np.random.rand(2, 1)


for i in range(n):
     x[i, 0] = np.dot(A[i, :], x0).item() + np.dot(B[i, :], u0).item()
    

for i in range(T-1):
     x[:, i+1] = np.dot(A, x[:, i]) + np.dot(B, u[:, i])
     
# Plot the results

time = np.dot(10, np.arange(0, T*T_s, T_s))
# time = np.arange(0, T*T_s, T_s)

plt.figure(figsize=(10, 6))
plt.plot(1,1)
plt.plot(time, x[0, :], label='x1')
plt.plot(time, x[1, :], label='x2')
plt.plot(time, x[2, :], label='x3')
plt.plot(time, x[3, :], label='x4')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.legend()
plt.title('State Response')
plt.tight_layout()
plt.show()