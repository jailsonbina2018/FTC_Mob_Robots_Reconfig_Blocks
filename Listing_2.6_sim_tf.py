# %% Import:
import numpy as np
import control as ct
import matplotlib.pyplot as plt


# %% Creating model:
num = np.array([2])
den = np.array([5, 1])
H = ct.tf(num, den)

# %% Defining signals:
t0 = 0
t1 = 20
dt = 0.01
nt = int(t1/dt) + 1 # Number of points of sim time
t = np.linspace(t0, t1, nt)
u = 1*np.ones(nt)

# %% Simulation:
(t, y) = ct.step_response(H, t, X0=0)

# %% Plotting:
plt.close('all')
fig_width_cm = 24
fig_height_cm = 18
plt.figure(1, figsize=(fig_width_cm/2.54, fig_height_cm/2.54))

plt.subplot(2, 1, 1)
plt.plot(t, y, 'blue')
plt.xlabel('t [s]')
plt.grid()
plt.legend(labels=('y',))

plt.subplot(2, 1, 2)
plt.plot(t, u, 'green')
plt.xlabel('t [s]')
plt.grid()
plt.legend(labels=('u',))
plt.savefig('sim_tf.pdf')
plt.show()


