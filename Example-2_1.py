import numpy as np
import control
# %% Creating the transfer function:
num = np.array([2])
den = np.array([5, 1])
H = control.tf(num, den)
# %% Displaying the transfer function:
print('H(s) = ', H)