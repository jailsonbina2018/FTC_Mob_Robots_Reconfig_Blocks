import numpy as np
import control as ctrl
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

# Assuming C and D matrices (here identity and zeros for simplicity)
C = np.eye(4)  # Output the states directly
D = np.zeros((4, 2))  # No direct transmission from input to output

# Create the state-space system
sys = ctrl.StateSpace(A, B, C, D)

# Define the time vector for the simulation
t = np.linspace(0, 10, 1000)

# Define the input vector for the simulation
# For simplicity, let's use a step input for both inputs
u = np.ones((1000, 2))

# Simulate the system
t, y, x = ctrl.forced_response(sys, T=t, U=u)

# Plot the results
plt.figure()
for i in range(y.shape[0]):
    plt.plot(t, y[i, :], label=f'State {i+1}')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.title('State Responses')
plt.legend()
plt.grid()
plt.show()
