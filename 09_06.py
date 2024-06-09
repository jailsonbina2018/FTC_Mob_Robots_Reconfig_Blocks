import numpy as np
import matplotlib.pyplot as plt
import control as ct

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

# Sampling time
dt = 0.1

# Create the continuous-time state-space model
sys_cont = ct.ss(A, B, C, D)

# Convert the continuous-time system to a discrete-time system
sys_disc = ct.c2d(sys_cont, dt)

# Define the input sequence (for example, a step input)
t = np.arange(0, 10, dt)
u = np.ones((2,len(t)))  # Step input for both inputs

# Define the initial state
x0 = np.zeros(4)

# Simulate the system response
t_out, y_out= ct.forced_response(sys_disc, T=t, U=u, X0=x0)

# Plot the states over time
plt.figure(figsize=(12, 8))
for i in range(y_out.shape[0]):
    plt.plot(t_out, y_out[i, :], label=f'State {i+1}')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.title('State Response Over Time')
plt.legend()
plt.grid(True)
plt.show()

